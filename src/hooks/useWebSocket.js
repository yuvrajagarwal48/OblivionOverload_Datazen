import { useEffect, useRef, useCallback } from 'react';
import useSimulationStore from '../store/simulationStore';
// eslint-disable-next-line no-unused-vars
import { WS_URL, USE_MOCK } from '../config';

/**
 * WebSocket hook — connects to /ws/simulation when simulation is running.
 * Server → Client only. Parses STATE_UPDATE, EVENT, METRICS_UPDATE.
 *
 * Skipped entirely when USE_MOCK is true (mock engine pushes data directly).
 */
export default function useWebSocket() {
  const wsRef = useRef(null);
  const reconnectTimer = useRef(null);
  const simStatus = useSimulationStore((s) => s.simStatus);
  const setStateUpdate = useSimulationStore((s) => s.setStateUpdate);
  const pushEvent = useSimulationStore((s) => s.pushEvent);
  const setMetrics = useSimulationStore((s) => s.setMetrics);
  const setSimStatus = useSimulationStore((s) => s.setSimStatus);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('[WS] Connected to simulation');
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);

        switch (msg.type) {
          case 'STATE_UPDATE':
            setStateUpdate(msg.payload);
            break;
          case 'EVENT':
            pushEvent(msg.payload);
            break;
          case 'METRICS_UPDATE':
            setMetrics(msg.payload);
            break;
          case 'SIM_COMPLETE':
            setSimStatus('done');
            break;
          default:
            console.warn('[WS] Unknown message type:', msg.type);
        }
      } catch (err) {
        console.error('[WS] Failed to parse message:', err);
      }
    };

    ws.onclose = (e) => {
      console.log('[WS] Disconnected', e.code, e.reason);
      wsRef.current = null;

      // Auto-reconnect if still running
      const currentStatus = useSimulationStore.getState().simStatus;
      if (currentStatus === 'running') {
        reconnectTimer.current = setTimeout(() => {
          console.log('[WS] Attempting reconnect...');
          connect();
        }, 2000);
      }
    };

    ws.onerror = (err) => {
      console.error('[WS] Error:', err);
      ws.close();
    };
  }, [setStateUpdate, pushEvent, setMetrics, setSimStatus]);

  const disconnect = useCallback(() => {
    if (reconnectTimer.current) {
      clearTimeout(reconnectTimer.current);
      reconnectTimer.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  // Connect when simulation starts, disconnect when it stops
  useEffect(() => {
    // In mock mode, data is pushed directly by the mock engine — no WebSocket needed
    if (USE_MOCK) return;

    if (simStatus === 'running') {
      connect();
    } else if (simStatus === 'idle') {
      disconnect();
    }

    return () => {
      disconnect();
    };
  }, [simStatus, connect, disconnect]);

  return { wsRef };
}
