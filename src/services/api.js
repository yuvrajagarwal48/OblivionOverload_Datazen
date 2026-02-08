/**
 * API Service Layer for FinSim-MAPPO
 *
 * Uses the LEGACY endpoints defined directly in main.py (no /api prefix).
 * These use the simpler SimulationState that works without bugs.
 *
 * Legacy endpoints:
 *   POST /simulation/init, /simulation/reset, /simulation/step, /simulation/run
 *   GET  /metrics, /metrics/risk, /metrics/bank/{id}
 *   GET  /network/topology
 *   GET  /scenarios, POST /scenarios/set, POST /scenarios/shock
 *   POST /what_if
 *   GET  /recommendations/{bankId}
 *   GET  /health
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
// SIMULATION CONTROL
// ═══════════════════════════════════════════════════════════════

/** POST /simulation/init — initialize environment + agents */
export const initSimulation = (config = {}) =>
  post('/simulation/init', {
    num_banks: config.num_banks ?? 5,
    episode_length: config.episode_length ?? 100,
    scenario: config.scenario ?? 'normal',
    seed: config.seed ?? null,
    banks: config.banks ?? undefined,
    ccps: config.ccps ?? undefined,
    market: config.market ?? undefined,
  });

/** POST /simulation/reset */
export const resetSimulation = () => post('/simulation/reset');

/**
 * POST /simulation/step — execute one step.
 * Response: { step, rewards, done, network_stats, market_state }
 */
export const stepSimulation = () => post('/simulation/step');

/**
 * POST /simulation/run?num_steps=N — run multiple steps at once.
 * Response: { steps_completed, total_rewards, final_step, network_stats }
 */
export const runSimulation = (numSteps = 100) =>
  post(`/simulation/run?num_steps=${numSteps}`);

// ═══════════════════════════════════════════════════════════════
// STATE & METRICS
// ═══════════════════════════════════════════════════════════════

/**
 * GET /metrics — full current state snapshot.
 * Response: { step, network, market, banks, scenario }
 */
export const getMetrics = () => get('/metrics');

/** GET /metrics/risk — comprehensive risk analysis (DebtRank etc). */
export const getRiskMetrics = () => get('/metrics/risk');

/**
 * GET /metrics/bank/{id} — detailed bank info.
 * Response: { bank_id, tier, status, balance_sheet, capital_ratio,
 *             is_solvent, is_liquid, excess_cash, centrality, neighbors, creditors, debtors }
 */
export const getBankDetails = (bankId) => get(`/metrics/bank/${bankId}`);

// ═══════════════════════════════════════════════════════════════
// BANK ENDPOINTS (from ai branch API)
// ═══════════════════════════════════════════════════════════════

/**
 * GET /api/bank/{bank_id}/history — time series of bank metrics.
 * Query params: start, end
 * Response: { bank_id, count, timesteps, data, series }
 */
export const getBankHistory = (bankId, start = 0, end = null) => {
  const params = new URLSearchParams({ start: start.toString() });
  if (end !== null) params.append('end', end.toString());
  return get(`/api/bank/${bankId}/history?${params}`);
};

/**
 * GET /api/bank/{bank_id}/transactions — transaction history.
 * Query params: limit, tx_type
 * Response: { bank_id, count, transactions, summary }
 */
export const getBankTransactions = (bankId, limit = 100, txType = null) => {
  const params = new URLSearchParams({ limit: limit.toString() });
  if (txType) params.append('tx_type', txType);
  return get(`/api/bank/${bankId}/transactions?${params}`);
};

/**
 * GET /api/bank/{bank_id}/exposures — detailed exposure breakdown.
 * Response: { bank_id, assets, liabilities, summary }
 */
export const getBankExposures = (bankId) => get(`/api/bank/${bankId}/exposures`);

/**
 * GET /api/bank/ — list all banks with summary.
 * Response: { count, banks, summary }
 */
export const listBanks = () => get('/api/bank/');

/**
 * GET /api/bank/stressed — get stressed/critical banks.
 * Response: { stressed, critical, defaulted, summary }
 */
export const getStressedBanks = () => get('/api/bank/stressed');

// ═══════════════════════════════════════════════════════════════
// NETWORK
// ═══════════════════════════════════════════════════════════════

/**
 * GET /network/topology — nodes + edges for graph visualization.
 * Response: { nodes, edges, stats }
 */
export const getNetworkTopology = () => get('/network/topology');

// ═══════════════════════════════════════════════════════════════
// SCENARIOS & SHOCKS
// ═══════════════════════════════════════════════════════════════

/** GET /scenarios */
export const listScenarios = () => get('/scenarios');

/** POST /scenarios/set */
export const setScenario = (scenarioName) =>
  post('/scenarios/set', { scenario_name: scenarioName });

/** POST /scenarios/shock */
export const applyShock = (shock) => post('/scenarios/shock', shock);

// ═══════════════════════════════════════════════════════════════
// WHAT-IF & RECOMMENDATIONS
// ═══════════════════════════════════════════════════════════════

/**
 * POST /api/whatif/analyze — comprehensive what-if transaction analysis.
 * Body: {
 *   transaction_type: string,  // 'loan_approval', 'margin_increase', 'borrow', 'sell_assets'
 *   initiator_id: int,
 *   counterparty_id: int (optional),
 *   amount: float,
 *   interest_rate: float (default 0.05),
 *   duration: int (default 10),
 *   collateral: float (default 0),
 *   horizon: int (default 10),
 *   num_simulations: int (default 20)
 * }
 * Response: {
 *   analysis_id, transaction, baseline, counterfactual, deltas,
 *   risk_assessment, recommendation, simulation_info
 * }
 */
export const analyzeTransaction = (payload) => post('/api/whatif/analyze', payload);

/**
 * GET /api/whatif/quick-check/{bank_id} — quick risk check.
 * Query params: amount, transaction_type
 * Response: { bank_id, current_state, post_transaction, assessment }
 */
export const quickRiskCheck = (bankId, amount, transactionType = 'loan_approval') => {
  const params = new URLSearchParams({
    amount: amount.toString(),
    transaction_type: transactionType
  });
  return get(`/api/whatif/quick-check/${bankId}?${params}`);
};

/** POST /what_if — LEGACY endpoint (basic action vector) */
export const whatIf = (payload) => post('/what_if', payload);

/** GET /recommendations/{bankId} */
export const getRecommendation = (bankId) => get(`/recommendations/${bankId}`);

/** GET /recommendations — all banks */
export const getAllRecommendations = () => get('/recommendations');

// ═══════════════════════════════════════════════════════════════
// HEALTH
// ═══════════════════════════════════════════════════════════════

/** GET /health */
export const healthCheck = () => get('/health');
