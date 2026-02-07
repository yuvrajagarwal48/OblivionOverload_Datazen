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
  const setNodeDecision = useSimulationStore((s) => s.setNodeDecision);

  // API-mode state
  const setApiLoading = useSimulationStore((s) => s.setApiLoading);
  const setApiError = useSimulationStore((s) => s.setApiError);
  const setBackendInitialized = useSimulationStore((s) => s.setBackendInitialized);
  const ingestBackendState = useSimulationStore((s) => s.ingestBackendState);
  const ingestEdges = useSimulationStore((s) => s.ingestEdges);
  const ingestSystemicRisk = useSimulationStore((s) => s.ingestSystemicRisk);
  const ingestTimeSeries = useSimulationStore((s) => s.ingestTimeSeries);
  const ingestStepResult = useSimulationStore((s) => s.ingestStepResult);
  const ingestMarketState = useSimulationStore((s) => s.ingestMarketState);

  // Polling ref
  const pollRef = useRef(null);

  // ─── Fetch full state from backend ───
  const fetchFullState = useCallback(async () => {
    try {
      // Fetch state, topology, analytics, time series in parallel
      const [stateRes, topoRes, riskRes, tsRes] = await Promise.allSettled([
        api.getSimulationState(),
        api.getNetworkTopology(),
        api.getSystemicRisk(),
        api.getTimeSeriesData('market_price,default_rate,avg_capital_ratio,liquidity_index'),
      ]);

      if (stateRes.status === 'fulfilled') {
        ingestBackendState(stateRes.value);
        if (stateRes.value.market) ingestMarketState(stateRes.value);
      }
      if (topoRes.status === 'fulfilled' && topoRes.value.edges) {
        ingestEdges(topoRes.value.edges);
      }
      if (riskRes.status === 'fulfilled') {
        ingestSystemicRisk(riskRes.value);
      }
      if (tsRes.status === 'fulfilled') {
        ingestTimeSeries(tsRes.value);
      }

      // Fetch events separately
      try {
        const eventsRes = await api.getSimulationEvents();
        if (eventsRes.events) {
          eventsRes.events.forEach((evt) => pushEvent(evt));
        }
      } catch { /* events optional */ }

    } catch (err) {
      console.error('[API] fetchFullState error:', err);
    }
  }, [ingestBackendState, ingestEdges, ingestSystemicRisk, ingestTimeSeries, ingestMarketState, pushEvent]);

  // ─── Poll loop: step + fetch state ───
  const startPolling = useCallback(() => {
    if (pollRef.current) return;

    const poll = async () => {
      try {
        // Step the simulation
        const stepResult = await api.stepSimulation(1);
        ingestStepResult(stepResult);

        // Check if done
        if (stepResult.is_done) {
          stopPolling();
          setSimStatus('done');
          // Final full fetch
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
    // Run immediately
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
    if (!scenario) {
      console.warn('[API] No scenario selected');
      return false;
    }

    setApiLoading(true);
    try {
      // Use 'normal' scenario for all frontend selections for now
      await api.initSimulation({
        num_banks: 30,
        episode_length: 100,
        scenario: 'normal',
      });
      setBackendInitialized(true);

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
  }, [selectedScenario, customConfig, setSimStatus, setApiLoading, setApiError, setBackendInitialized, fetchFullState, startPolling]);

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
      const result = await api.stepSimulation(1);
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
