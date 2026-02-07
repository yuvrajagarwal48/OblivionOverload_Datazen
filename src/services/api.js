/**
 * API Service Layer for FinSim-MAPPO
 *
 * All REST calls to the FastAPI backend.
 * Routes match: api/routes/{simulation, banks, analytics, market, infrastructure, whatif}.py
 * All routes are prefixed with /api (set in main.py: app.include_router(..., prefix="/api"))
 */

import { API_BASE_URL } from '../config';

// ─── Helper ───

async function request(path, options = {}) {
  const url = `${API_BASE_URL}${path}`;
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json();
}

const get = (path) => request(path);
const post = (path, body) =>
  request(path, { method: 'POST', body: body != null ? JSON.stringify(body) : undefined });

// ═══════════════════════════════════════════════════════════════
// SIMULATION  (/api/simulation/*)
// ═══════════════════════════════════════════════════════════════

/** POST /api/simulation/init */
export const initSimulation = (config = {}) =>
  post('/api/simulation/init', {
    num_banks: config.num_banks ?? 30,
    episode_length: config.episode_length ?? 100,
    scenario: config.scenario ?? 'normal',
    num_exchanges: config.num_exchanges ?? 2,
    num_ccps: config.num_ccps ?? 1,
    seed: config.seed ?? null,
  });

/** POST /api/simulation/reset */
export const resetSimulation = () => post('/api/simulation/reset');

/** POST /api/simulation/step — single or multi-step */
export const stepSimulation = (numSteps = 1) =>
  post('/api/simulation/step', { num_steps: numSteps, capture_state: true });

/** POST /api/simulation/run?num_steps=N */
export const runSimulation = (numSteps = 100) =>
  post(`/api/simulation/run?num_steps=${numSteps}`);

/** POST /api/simulation/shock */
export const applyShock = (shock) => post('/api/simulation/shock', shock);

/** GET /api/simulation/status */
export const getSimulationStatus = () => get('/api/simulation/status');

/** GET /api/simulation/state — full snapshot */
export const getSimulationState = () => get('/api/simulation/state');

/** GET /api/simulation/history?start=&end=&fields= */
export const getSimulationHistory = (start = 0, end = null, fields = null) => {
  const params = new URLSearchParams({ start });
  if (end != null) params.append('end', end);
  if (fields) params.append('fields', fields);
  return get(`/api/simulation/history?${params}`);
};

/** GET /api/simulation/history/timeseries?metrics=... */
export const getTimeSeriesData = (
  metrics = 'market_price,default_rate,avg_capital_ratio,total_exposure,liquidity_index'
) => get(`/api/simulation/history/timeseries?metrics=${encodeURIComponent(metrics)}`);

/** GET /api/simulation/history/graphs — pre-formatted chart data */
export const getGraphData = () => get('/api/simulation/history/graphs');

/** GET /api/simulation/history/clearing */
export const getClearingHistory = () => get('/api/simulation/history/clearing');

/** GET /api/simulation/history/events?event_type= */
export const getSimulationEvents = (eventType = null) => {
  const params = eventType ? `?event_type=${eventType}` : '';
  return get(`/api/simulation/history/events${params}`);
};

// ═══════════════════════════════════════════════════════════════
// BANKS  (/api/bank/*)
// ═══════════════════════════════════════════════════════════════

/** GET /api/bank/ — list all banks */
export const listBanks = () => get('/api/bank/');

/** GET /api/bank/{id} — full details */
export const getBankDetails = (bankId) => get(`/api/bank/${bankId}`);

/** GET /api/bank/{id}/history */
export const getBankHistory = (bankId, start = 0, end = null) => {
  const params = new URLSearchParams({ start });
  if (end != null) params.append('end', end);
  return get(`/api/bank/${bankId}/history?${params}`);
};

/** GET /api/bank/{id}/transactions */
export const getBankTransactions = (bankId, limit = 100, txType = null) => {
  const params = new URLSearchParams({ limit });
  if (txType) params.append('tx_type', txType);
  return get(`/api/bank/${bankId}/transactions?${params}`);
};

// ═══════════════════════════════════════════════════════════════
// ANALYTICS  (/api/analytics/*)
// ═══════════════════════════════════════════════════════════════

/** GET /api/analytics/systemic-risk */
export const getSystemicRisk = () => get('/api/analytics/systemic-risk');

/** GET /api/analytics/debtrank */
export const getDebtRank = () => get('/api/analytics/debtrank');

/** GET /api/analytics/network */
export const getNetworkAnalytics = () => get('/api/analytics/network');

// ═══════════════════════════════════════════════════════════════
// MARKET  (/api/market/*)
// ═══════════════════════════════════════════════════════════════

/** GET /api/market/state */
export const getMarketState = () => get('/api/market/state');

// ═══════════════════════════════════════════════════════════════
// INFRASTRUCTURE  (/api/infrastructure/*)
// ═══════════════════════════════════════════════════════════════

/** GET /api/infrastructure/status */
export const getInfrastructureStatus = () => get('/api/infrastructure/status');

// ═══════════════════════════════════════════════════════════════
// WHAT-IF  (/api/whatif/*)
// ═══════════════════════════════════════════════════════════════

/** POST /api/whatif/evaluate */
export const evaluateWhatIf = (payload) => post('/api/whatif/evaluate', payload);

// ═══════════════════════════════════════════════════════════════
// LEGACY ENDPOINTS (defined directly in main.py, no /api prefix)
// ═══════════════════════════════════════════════════════════════

/** GET /health */
export const healthCheck = () => get('/health');

/** GET /metrics */
export const getMetrics = () => get('/metrics');

/** GET /metrics/risk */
export const getRiskMetrics = () => get('/metrics/risk');

/** GET /metrics/bank/{id} */
export const getLegacyBankMetrics = (bankId) => get(`/metrics/bank/${bankId}`);

/** GET /network/topology */
export const getNetworkTopology = () => get('/network/topology');

/** POST /what_if (legacy) */
export const legacyWhatIf = (payload) => post('/what_if', payload);

/** GET /recommendations/{bankId} */
export const getRecommendation = (bankId) => get(`/recommendations/${bankId}`);

/** GET /scenarios */
export const listScenarios = () => get('/scenarios');

/** POST /scenarios/set */
export const setScenario = (scenarioName) =>
  post('/scenarios/set', { scenario_name: scenarioName });
