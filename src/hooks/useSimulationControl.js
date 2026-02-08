import { useCallback, useRef } from 'react';
import useSimulationStore from '../store/simulationStore';
import { USE_MOCK, POLL_INTERVAL } from '../config';
import { getMockSimulation } from '../mock/mockSimulation';
import * as api from '../services/api';

/**
 * Simulation lifecycle controller.
 *
 * USE_MOCK=true  → mock engine (no backend)
 * USE_MOCK=false → real FastAPI backend (legacy endpoints) with polling loop
 *
 * Legacy endpoints used:
 *   POST /simulation/init, /simulation/step, /simulation/reset
 *   GET  /metrics (full state), /metrics/risk (DebtRank), /network/topology (graph)
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
  const setNodeDecision = useSimulationStore((s) => s.setNodeDecision);

  // API-mode state
  const setApiLoading = useSimulationStore((s) => s.setApiLoading);
  const setApiError = useSimulationStore((s) => s.setApiError);
  const setBackendInitialized = useSimulationStore((s) => s.setBackendInitialized);
  const ingestMetrics = useSimulationStore((s) => s.ingestMetrics);
  const ingestEdges = useSimulationStore((s) => s.ingestEdges);
  const ingestTopologyNodes = useSimulationStore((s) => s.ingestTopologyNodes);
  const ingestStepResult = useSimulationStore((s) => s.ingestStepResult);
  const ingestRiskMetrics = useSimulationStore((s) => s.ingestRiskMetrics);

  // Polling ref
  const pollRef = useRef(null);

  // ─── Fetch full state from backend (legacy endpoints) ───
  const fetchFullState = useCallback(async () => {
    try {
      const [metricsRes, topoRes, riskRes] = await Promise.allSettled([
        api.getMetrics(),
        api.getNetworkTopology(),
        api.getRiskMetrics(),
      ]);

      if (metricsRes.status === 'fulfilled') {
        ingestMetrics(metricsRes.value);
      }
      if (topoRes.status === 'fulfilled') {
        if (topoRes.value.edges) ingestEdges(topoRes.value.edges);
        // Ingest topology nodes (may include CCP nodes)
        if (topoRes.value.nodes) {
          ingestTopologyNodes(topoRes.value.nodes);
          // If metrics fetch failed, also populate basic bank data from topology
          if (metricsRes.status !== 'fulfilled') {
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
  }, [ingestMetrics, ingestEdges, ingestTopologyNodes, ingestRiskMetrics]);

  // ─── Poll loop: step + fetch state ───
  const startPolling = useCallback(() => {
    if (pollRef.current) return;

    const poll = async () => {
      try {
        // Step the simulation
        const stepResult = await api.stepSimulation();
        ingestStepResult(stepResult);

        // Check if done
        if (stepResult.done) {
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
      // Build config from manual setup or use defaults
      const numBanksFromConfig = manualConfig.banks.length > 0 ? manualConfig.banks.length : 10;
      
      const initConfig = useManualSetup ? {
        num_banks: numBanksFromConfig,
        episode_length: manualConfig.market.episode_length || 100,
        scenario: 'custom',
        banks: manualConfig.banks.length > 0 ? manualConfig.banks : undefined,
        ccps: manualConfig.ccps.length > 0 ? manualConfig.ccps : undefined,
        market: manualConfig.market,
      } : {
        num_banks: 5,
        episode_length: 100,
        scenario: 'normal',
      };

      console.log('[API] Initializing with config:', initConfig);

      // Initialize simulation via legacy endpoint
      await api.initSimulation(initConfig);
      setBackendInitialized(true);

      // Fetch initial state
      await fetchFullState();

      // If manual setup has CCPs, inject them as nodes if backend didn't
      if (useManualSetup && manualConfig.ccps.length > 0) {
        const currentNodes = useSimulationStore.getState().nodes;
        const ccpNodes = manualConfig.ccps
          .filter(ccp => !currentNodes.find(n => String(n.id) === String(ccp.id)))
          .map(ccp => ({
            id: ccp.id,
            label: ccp.name || `CCP-${ccp.id}`,
            node_type: 'ccp',
            tier: 0,
            capital_ratio: 1.0,
            stress: 0,
            status: 'active',
          }));
        if (ccpNodes.length > 0) {
          ingestTopologyNodes([...currentNodes, ...ccpNodes]);
        }
      }

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
  }, [selectedScenario, useManualSetup, manualConfig, setSimStatus, setApiLoading, setApiError, setBackendInitialized, fetchFullState, startPolling]);

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
