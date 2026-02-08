/**
 * API Service Layer for FinSim-MAPPO
 *
 * Unified layer:
 *   - New modular: /api/simulation/*, /api/bank/*, /api/bank/registry
 *   - Legacy fallbacks: /metrics, /metrics/risk, /network/topology, /what_if
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
// BANK REGISTRY (real banks from RBI data — no init required)
// ═══════════════════════════════════════════════════════════════

/** GET /api/bank/registry — all 46 real Indian banks with metadata */
export const getBankRegistry = () => get('/api/bank/registry');

/** GET /api/bank/registry/search?q=... — search banks by name */
export const searchBankRegistry = (query) =>
  get(`/api/bank/registry/search?q=${encodeURIComponent(query)}`);

/** GET /api/bank/registry/{id} — single bank detail from registry */
export const getRegistryBank = (bankId) => get(`/api/bank/registry/${bankId}`);

// ═══════════════════════════════════════════════════════════════
// SIMULATION CONTROL
// ═══════════════════════════════════════════════════════════════

/**
 * POST /api/simulation/init — initialize with real/synthetic banks.
 * Supports: bank_ids (real), bank_names (real), synthetic_count, num_banks (random fallback)
 */
export const initSimulation = (config = {}) =>
  post('/api/simulation/init', {
    bank_ids: config.bank_ids ?? undefined,
    bank_names: config.bank_names ?? undefined,
    synthetic_count: config.synthetic_count ?? undefined,
    synthetic_stress: config.synthetic_stress ?? undefined,
    num_banks: config.num_banks ?? 10,
    episode_length: config.episode_length ?? 100,
    scenario: config.scenario ?? 'normal',
    seed: config.seed ?? null,
  });

/** POST /simulation/reset (legacy) */
export const resetSimulation = () => post('/simulation/reset');

/** POST /api/simulation/step — execute one step */
export const stepSimulation = () => post('/api/simulation/step');

/** POST /simulation/run?num_steps=N (legacy) */
export const runSimulation = (numSteps = 100) =>
  post(`/simulation/run?num_steps=${numSteps}`);

// ═══════════════════════════════════════════════════════════════
// STATE & METRICS
// ═══════════════════════════════════════════════════════════════

/** GET /api/simulation/state — full state snapshot (banks with names, market, network) */
export const getSimulationState = () => get('/api/simulation/state');

/** GET /api/simulation/status — quick status summary */
export const getSimulationStatus = () => get('/api/simulation/status');

/** GET /metrics — legacy full state (fallback) */
export const getMetrics = () => get('/metrics');

/** GET /api/analytics/systemic-risk — comprehensive risk analysis (DebtRank etc.) */
export const getRiskMetrics = () =>
  get('/api/analytics/systemic-risk').catch(() => get('/metrics/risk'));

/** GET /api/analytics/debtrank — individual bank DebtRank rankings */
export const getDebtRank = () => get('/api/analytics/debtrank');

/** GET /api/analytics/credit-risk — system-wide credit risk */
export const getCreditRisk = () => get('/api/analytics/credit-risk');

/** GET /metrics/bank/{id} — legacy detailed bank info */
export const getLegacyBankDetails = (bankId) => get(`/metrics/bank/${bankId}`);

// ═══════════════════════════════════════════════════════════════
// BANK ENDPOINTS
// ═══════════════════════════════════════════════════════════════

/** GET /api/bank/ — all banks with names, metadata, risk scores */
export const listBanks = () => get('/api/bank/');

/** GET /api/bank/{id} — full bank detail (balance sheet, credit risk, network, margins) */
export const getBankDetails = (bankId) => get(`/api/bank/${bankId}`);

/** GET /api/bank/{id}/history — time series */
export const getBankHistory = (bankId, start = 0, end = null) => {
  const params = new URLSearchParams({ start: start.toString() });
  if (end !== null) params.append('end', end.toString());
  return get(`/api/bank/${bankId}/history?${params}`);
};

/** GET /api/bank/{id}/transactions */
export const getBankTransactions = (bankId, limit = 100, txType = null) => {
  const params = new URLSearchParams({ limit: limit.toString() });
  if (txType) params.append('tx_type', txType);
  return get(`/api/bank/${bankId}/transactions?${params}`);
};

/** GET /api/bank/{id}/exposures */
export const getBankExposures = (bankId) => get(`/api/bank/${bankId}/exposures`);

/** GET /api/bank/stressed */
export const getStressedBanks = () => get('/api/bank/stressed');

// ═══════════════════════════════════════════════════════════════
// NETWORK
// ═══════════════════════════════════════════════════════════════

/** GET /api/analytics/network/graph — nodes + edges for graph visualization */
export const getNetworkTopology = () =>
  get('/api/analytics/network/graph').catch(() => get('/network/topology'));

// ═══════════════════════════════════════════════════════════════
// SCENARIOS & SHOCKS (legacy)
// ═══════════════════════════════════════════════════════════════

export const listScenarios = () => get('/scenarios');
export const setScenario = (scenarioName) =>
  post('/scenarios/set', { scenario_name: scenarioName });
export const applyShock = (shock) => post('/scenarios/shock', shock);

// ═══════════════════════════════════════════════════════════════
// WHAT-IF & RECOMMENDATIONS
// ═══════════════════════════════════════════════════════════════

/** POST /api/whatif/analyze — comprehensive what-if analysis */
export const analyzeTransaction = (payload) => post('/api/whatif/analyze', payload);

/** GET /api/whatif/quick-check/{bank_id} */
export const quickRiskCheck = (bankId, amount, transactionType = 'loan_approval') => {
  const params = new URLSearchParams({
    amount: amount.toString(),
    transaction_type: transactionType
  });
  return get(`/api/whatif/quick-check/${bankId}?${params}`);
};

/** POST /what_if — legacy action vector */
export const whatIf = (payload) => post('/what_if', payload);

/** GET /recommendations/{bankId} */
export const getRecommendation = (bankId) => get(`/recommendations/${bankId}`);

// ═══════════════════════════════════════════════════════════════
// AI INSIGHTS (GenAI analysis & reports)
// ═══════════════════════════════════════════════════════════════

/** POST /api/ai-insights/generate — generate comprehensive AI analysis */
export const generateAIAnalysis = (config = {}) => post('/api/ai-insights/generate', config);

/** GET /api/ai-insights/report/json — latest analysis as structured JSON */
export const getAIReport = () => get('/api/ai-insights/report/json');

/** GET /api/ai-insights/report/text — latest analysis as plain text */
export const getAIReportText = () =>
  request('/api/ai-insights/report/text').then(r => r).catch(async (err) => {
    // text endpoint returns plain text, re-fetch as text
    const res = await fetch(`${API_BASE_URL}/api/ai-insights/report/text`);
    if (!res.ok) throw new Error(`API ${res.status}`);
    return res.text();
  });

/** GET /api/ai-insights/metrics — raw aggregated metrics */
export const getAIMetrics = () => get('/api/ai-insights/metrics');

/** GET /api/ai-insights/risk-scores — risk assessment scores (0-100) */
export const getAIRiskScores = () => get('/api/ai-insights/risk-scores');

/** GET /api/ai-insights/recommendations — policy recommendations */
export const getAIRecommendations = () => get('/api/ai-insights/recommendations');

/** GET /api/ai-insights/bank-insights — per-bank AI insights */
export const getAIBankInsights = () => get('/api/ai-insights/bank-insights');

/** GET /health */
export const healthCheck = () => get('/health');
