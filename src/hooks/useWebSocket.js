/**
 * WebSocket hook — currently a no-op.
 *
 * The backend is purely REST-based. All real-time data is fetched via
 * polling in useSimulationControl.js and the analytics panels.
 *
 * If a WebSocket endpoint is added to the backend in the future,
 * this hook can be expanded to subscribe to push updates.
 */
export default function useWebSocket() {
  // No-op — backend has no WebSocket endpoint.
  // All data flows through REST polling.
  return {};
}
