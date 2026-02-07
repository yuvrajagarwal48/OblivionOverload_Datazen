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
    num_banks: config.num_banks ?? 30,
    episode_length: config.episode_length ?? 100,
    scenario: config.scenario ?? 'normal',
    seed: config.seed ?? null,
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

/** POST /what_if */
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
