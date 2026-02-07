import { useCallback } from 'react';
import useSimulationStore from '../store/simulationStore';
import { API_BASE_URL, USE_MOCK } from '../config';
import { getMockSimulation } from '../mock/mockSimulation';

/**
 * REST control hooks for simulation lifecycle.
 * POST /control/{play,pause,step,reset}
 *
 * When USE_MOCK is true, uses the mock simulation engine instead of REST.
 */
export default function useSimulationControl() {
  const setSimStatus = useSimulationStore((s) => s.setSimStatus);
  const setStateUpdate = useSimulationStore((s) => s.setStateUpdate);
  const pushEvent = useSimulationStore((s) => s.pushEvent);
  const setMetrics = useSimulationStore((s) => s.setMetrics);
  const resetAll = useSimulationStore((s) => s.resetAll);
  const selectedScenario = useSimulationStore((s) => s.selectedScenario);
  const customConfig = useSimulationStore((s) => s.customConfig);
  const setNodeDecision = useSimulationStore((s) => s.setNodeDecision);

  const postControl = useCallback(async (action, body = null) => {
    try {
      const res = await fetch(`${API_BASE_URL}/control/${action}`, {
        method: 'POST',
        headers: body ? { 'Content-Type': 'application/json' } : {},
        body: body ? JSON.stringify(body) : null,
      });
      if (!res.ok) {
        console.error(`[Control] ${action} failed:`, res.status, await res.text());
        return false;
      }
      return true;
    } catch (err) {
      console.error(`[Control] ${action} error:`, err);
      return false;
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
      // Track the decision on relevant nodes
      if (event.from) setNodeDecision(event.from, event.event_type);
      if (event.bank) setNodeDecision(event.bank, event.event_type);
    };
    mock.onMetricsUpdate = (payload) => setMetrics(payload);
    mock.onComplete = () => setSimStatus('done');

    mock.start(scenario, scenario === 'custom' ? customConfig : null);
    setSimStatus('running');
    console.log('[Mock] Simulation started:', scenario);
    return true;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedScenario, customConfig, setSimStatus, setStateUpdate, pushEvent, setMetrics]);

  // ─── Play ───
  const play = useCallback(async () => {
    if (USE_MOCK) return playMock();

    const scenario = selectedScenario;
    if (!scenario) {
      console.warn('[Control] No scenario selected');
      return false;
    }

    const body = {
      scenario,
      ...(scenario === 'custom' ? { config: customConfig } : {}),
    };

    const ok = await postControl('play', body);
    if (ok) setSimStatus('running');
    return ok;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedScenario, customConfig, postControl, setSimStatus, playMock]);

  // ─── Pause ───
  const pause = useCallback(async () => {
    if (USE_MOCK) {
      getMockSimulation().pause();
      setSimStatus('paused');
      return true;
    }
    const ok = await postControl('pause');
    if (ok) setSimStatus('paused');
    return ok;
  }, [postControl, setSimStatus]);

  // ─── Step ───
  const step = useCallback(async () => {
    if (USE_MOCK) {
      getMockSimulation().stepOnce();
      return true;
    }
    const ok = await postControl('step');
    return ok;
  }, [postControl]);

  // ─── Reset ───
  const reset = useCallback(async () => {
    if (USE_MOCK) {
      getMockSimulation().reset();
      resetAll();
      return true;
    }
    const ok = await postControl('reset');
    if (ok) resetAll();
    return ok;
  }, [postControl, resetAll]);

  return { play, pause, step, reset };
}
