import { useCallback, useRef } from 'react';
import useSimulationStore from '../store/simulationStore';
import { USE_MOCK, POLL_INTERVAL } from '../config';
import { getMockSimulation } from '../mock/mockSimulation';
import * as api from '../services/api';

/**
 * Simulation lifecycle controller.
 *
 * USE_MOCK=true  → mock engine (no backend)
 * USE_MOCK=false → real FastAPI backend with polling loop
 *
 * Uses new modular endpoints:
 *   POST /api/simulation/init, /api/simulation/step
 *   GET  /api/simulation/state (full state), /metrics/risk (DebtRank), /network/topology (graph)
 */
export default function useSimulationControl() {
  const setSimStatus = useSimulationStore((s) => s.setSimStatus);
  const setStateUpdate = useSimulationStore((s) => s.setStateUpdate);
  const pushEvent = useSimulationStore((s) => s.pushEvent);
  const setMetrics = useSimulationStore((s) => s.setMetrics);
  const setAdvancedMetrics = useSimulationStore((s) => s.setAdvancedMetrics);
  const pushTimeSeriesData = useSimulationStore((s) => s.pushTimeSeriesData);
  const setBankDebtRanks = useSimulationStore((s) => s.setBankDebtRanks);
  const resetAll = useSimulationStore((s) => s.resetAll);
  const selectedScenario = useSimulationStore((s) => s.selectedScenario);
  const customConfig = useSimulationStore((s) => s.customConfig);
  const manualConfig = useSimulationStore((s) => s.manualConfig);
  const useManualSetup = useSimulationStore((s) => s.useManualSetup);
  const selectedRealBanks = useSimulationStore((s) => s.selectedRealBanks);
  const setNodeDecision = useSimulationStore((s) => s.setNodeDecision);

  // API-mode state
  const setApiLoading = useSimulationStore((s) => s.setApiLoading);
  const setApiError = useSimulationStore((s) => s.setApiError);
  const setBackendInitialized = useSimulationStore((s) => s.setBackendInitialized);
  const ingestMetrics = useSimulationStore((s) => s.ingestMetrics);
  const ingestSimulationState = useSimulationStore((s) => s.ingestSimulationState);
  const ingestEdges = useSimulationStore((s) => s.ingestEdges);
  const ingestTopologyNodes = useSimulationStore((s) => s.ingestTopologyNodes);
  const ingestStepResult = useSimulationStore((s) => s.ingestStepResult);
  const ingestRiskMetrics = useSimulationStore((s) => s.ingestRiskMetrics);
  const setBankNameMap = useSimulationStore((s) => s.setBankNameMap);

  // Polling ref
  const pollRef = useRef(null);

  // ─── Fetch full state from backend ───
  const fetchFullState = useCallback(async () => {
    try {
      // Primary: new /api/simulation/state (has bank names, exchange/CCP data)
      // Fallback: legacy /metrics if new endpoint fails
      const [stateRes, topoRes, riskRes] = await Promise.allSettled([
        api.getSimulationState().catch(() => api.getMetrics()),
        api.getNetworkTopology(),
        api.getRiskMetrics(),
      ]);

      if (stateRes.status === 'fulfilled') {
        const data = stateRes.value;
        // New format has 'timestep' and 'network_stats', legacy has 'step' and 'network'
        if (data.network_stats !== undefined) {
          ingestSimulationState(data);
        } else {
          ingestMetrics(data);
        }
      }
      if (topoRes.status === 'fulfilled') {
        if (topoRes.value.edges) ingestEdges(topoRes.value.edges);
        if (topoRes.value.nodes) {
          ingestTopologyNodes(topoRes.value.nodes);
          if (stateRes.status !== 'fulfilled') {
            ingestMetrics({ banks: Object.fromEntries(topoRes.value.nodes.map(n => [n.id, n])), step: 0 });
          }
        }
      }
      if (riskRes.status === 'fulfilled') {
        ingestRiskMetrics(riskRes.value);
      }
    } catch (err) {
      console.error('[API] fetchFullState error:', err);
    }
  }, [ingestMetrics, ingestSimulationState, ingestEdges, ingestTopologyNodes, ingestRiskMetrics]);

  // ─── Poll loop: step + fetch state ───
  const startPolling = useCallback(() => {
    if (pollRef.current) return;

    const poll = async () => {
      try {
        // Step the simulation
        const stepResult = await api.stepSimulation();
        ingestStepResult(stepResult);

        // Check if done (new: is_done, legacy: done)
        if (stepResult.is_done || stepResult.done) {
          stopPolling();
          setSimStatus('done');
          await fetchFullState();
          return;
        }

        // Fetch full state after step
        await fetchFullState();

      } catch (err) {
        console.error('[API] Polling error:', err);
        setApiError(err.message);
        stopPolling();
        setSimStatus('paused');
      }
    };

    pollRef.current = setInterval(poll, POLL_INTERVAL);
    poll();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fetchFullState, ingestStepResult, setSimStatus, setApiError]);

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  // ─── Mock-mode play ───
  const playMock = useCallback(() => {
    const scenario = selectedScenario;
    if (!scenario) {
      console.warn('[Mock] No scenario selected');
      return false;
    }

    const mock = getMockSimulation();
    mock.onStateUpdate = (payload) => setStateUpdate(payload);
    mock.onEvent = (event) => {
      pushEvent(event);
      if (event.from) setNodeDecision(event.from, event.event_type);
      if (event.bank) setNodeDecision(event.bank, event.event_type);
    };
    mock.onMetricsUpdate = (payload) => setMetrics(payload);
    mock.onAdvancedMetricsUpdate = (payload) => setAdvancedMetrics(payload);
    mock.onTimeSeriesUpdate = (data) => pushTimeSeriesData(data);
    mock.onDebtRankUpdate = (payload) => setBankDebtRanks(payload);
    mock.onComplete = () => setSimStatus('done');

    mock.start(scenario, scenario === 'custom' ? customConfig : null);
    setSimStatus('running');
    console.log('[Mock] Simulation started:', scenario);
    return true;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedScenario, customConfig, setSimStatus, setStateUpdate, pushEvent, setMetrics]);

  // ─── API-mode play ───
  const playAPI = useCallback(async () => {
    const scenario = selectedScenario;
    if (!useManualSetup && !scenario) {
      console.warn('[API] No scenario selected');
      return false;
    }

    setApiLoading(true);
    try {
      let initConfig;

      if (useManualSetup && selectedRealBanks.length > 0) {
        // ── Real bank mode: use bank_ids from registry ──
        initConfig = {
          bank_ids: selectedRealBanks,
          num_banks: selectedRealBanks.length,
          episode_length: manualConfig.market.episode_length || 100,
          scenario: scenario || 'normal',
        };
      } else if (useManualSetup) {
        // ── Manual config mode (no real banks selected) ──
        initConfig = {
          num_banks: manualConfig.banks.length > 0 ? manualConfig.banks.length : 10,
          episode_length: manualConfig.market.episode_length || 100,
          scenario: 'custom',
        };
      } else {
        // ── Scenario-only mode (random banks) ──
        initConfig = {
          num_banks: 10,
          episode_length: 100,
          scenario: scenario || 'normal',
        };
      }

      console.log('[API] Initializing with config:', initConfig);

      // Initialize simulation
      const initResult = await api.initSimulation(initConfig);
      setBackendInitialized(true);

      // Build name map from init response bank list
      if (initResult.banks && Array.isArray(initResult.banks)) {
        const nameMap = {};
        initResult.banks.forEach((b) => {
          nameMap[String(b.id)] = b.name;
        });
        setBankNameMap(nameMap);
      }

      // Fetch initial state
      await fetchFullState();

      setSimStatus('running');
      setApiLoading(false);

      // Start polling loop
      startPolling();

      console.log('[API] Simulation started:', scenario);
      return true;
    } catch (err) {
      console.error('[API] Init failed:', err);
      setApiError(err.message);
      setApiLoading(false);
      return false;
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedScenario, useManualSetup, selectedRealBanks, manualConfig, setSimStatus, setApiLoading, setApiError, setBackendInitialized, setBankNameMap, fetchFullState, startPolling]);

  // ─── Play ───
  const play = useCallback(async () => {
    if (USE_MOCK) return playMock();
    return playAPI();
  }, [playMock, playAPI]);

  // ─── Pause ───
  const pause = useCallback(async () => {
    if (USE_MOCK) {
      getMockSimulation().pause();
      setSimStatus('paused');
      return true;
    }
    stopPolling();
    setSimStatus('paused');
    return true;
  }, [setSimStatus, stopPolling]);

  // ─── Step ───
  const step = useCallback(async () => {
    if (USE_MOCK) {
      getMockSimulation().stepOnce();
      return true;
    }
    try {
      const result = await api.stepSimulation();
      ingestStepResult(result);
      await fetchFullState();
      return true;
    } catch (err) {
      console.error('[API] Step error:', err);
      setApiError(err.message);
      return false;
    }
  }, [ingestStepResult, fetchFullState, setApiError]);

  // ─── Reset ───
  const reset = useCallback(async () => {
    stopPolling();
    if (USE_MOCK) {
      getMockSimulation().reset();
      resetAll();
      return true;
    }
    try {
      await api.resetSimulation();
      resetAll();
      return true;
    } catch (err) {
      console.error('[API] Reset error:', err);
      resetAll();
      return true;
    }
  }, [resetAll, stopPolling]);

  return { play, pause, step, reset };
}
