import { create } from 'zustand';
import { BANK_HISTORY_SIZE } from '../config';

const useSimulationStore = create((set, get) => ({
  // ─── Simulation State ───
  nodes: [],
  edges: [],
  events: [],
  metrics: {
    liquidity: null,
    default_rate: null,
    equilibrium_score: null,
    volatility: null,
  },
  timestep: 0,

  // ─── Advanced Metrics ───
  advancedMetrics: {
    // Systemic Risk
    aggregate_debtrank: null,
    cascade_depth: null,
    cascade_potential: null,
    critical_banks: [],
    systemic_risk_index: null,
    
    // Network
    network_density: null,
    clustering_coefficient: null,
    avg_path_length: null,
    concentration_index: null,
    
    // Market
    market_price: null,
    interest_rate: null,
    market_stress_regime: 'NORMAL',
    liquidity_index: null,
    
    // Clearing
    total_losses: null,
    clearing_iterations: null,
    avg_recovery_rate: null,
  },

  // ─── Time Series History ───
  timeSeriesHistory: {
    market_prices: [],
    interest_rates: [],
    liquidity_indices: [],
    default_rates: [],
    system_capital_ratios: [],
  },

  // ─── Per-Bank DebtRank ───
  bankDebtRanks: {}, // { [bankId]: debtrank_value }

  // ─── Auth State ───
  isAuthenticated: false,
  currentBankId: null,
  currentBankData: null,
  restrictedMode: false,

  // ─── API State ───
  apiLoading: false,
  apiError: null,
  backendInitialized: false, // tracks if /simulation/init has been called

  // ─── UI State ───
  showLanding: true,
  simStatus: 'idle', // idle | running | paused | done
  selectedScenario: null,
  customConfig: {
    shockType: 'liquidity',
    severity: 'medium',
    target: 'tier1',
    policy: 'stable',
  },
  selectedBanks: [],
  bankViewOpen: false,

  // ─── View State ───
  activeView: 'overview', // 'overview' | 'network' | 'analytics' | 'inspector'

  // ─── Node Decisions (last action per node) ───
  nodeDecisions: {}, // { [nodeId]: 'LEND' | 'HOARD' | 'FIRE_SALE' | ... }

  // ─── Bank History (rolling buffer for time-series) ───
  bankHistory: {}, // { [bankId]: [{ timestep, capital_ratio, stress, status }, ...] }

  // ─── Layout tracking ───
  layoutComputed: false,

  // ─── Actions: API State ───
  setApiLoading: (val) => set({ apiLoading: val }),
  setApiError: (err) => set({ apiError: err }),
  clearApiError: () => set({ apiError: null }),
  setBackendInitialized: (val) => set({ backendInitialized: val }),

  // ─── Actions: Ingest full backend state snapshot ───
  ingestBackendState: (data) => {
    // data comes from GET /api/simulation/state
    // { timestep, market, network_stats, banks, exchanges, ccps }
    const prevHistory = get().bankHistory;
    const newHistory = { ...prevHistory };

    // Convert banks dict → nodes array
    const nodes = Object.values(data.banks || {}).map((b) => {
      const id = String(b.bank_id);
      const entry = {
        timestep: data.timestep,
        capital_ratio: b.capital_ratio,
        stress: b.stress || 0,
        status: b.status,
      };
      if (!newHistory[id]) newHistory[id] = [];
      newHistory[id] = [...newHistory[id].slice(-(49)), entry];

      return {
        id,
        tier: b.tier,
        capital_ratio: b.capital_ratio,
        stress: b.stress || 0,
        status: b.status,
        cash: b.cash,
        total_assets: b.total_assets,
        total_liabilities: b.total_liabilities,
        equity: b.equity,
        illiquid_assets: b.illiquid_assets || 0,
        external_liabilities: b.external_liabilities || 0,
        interbank_assets: b.interbank_assets || {},
        interbank_liabilities: b.interbank_liabilities || {},
        debtrank: b.debtrank || 0,
        last_updated_timestep: data.timestep,
      };
    });

    set({
      timestep: data.timestep,
      nodes,
      bankHistory: newHistory,
    });
  },

  // ─── Ingest network edges (from /network/topology or /api/simulation/state) ───
  ingestEdges: (edgeList) => {
    const edges = edgeList.map((e) => ({
      source: String(e.source),
      target: String(e.target),
      weight: e.weight,
      type: e.type || 'credit',
    }));
    set({ edges });
  },

  // ─── Ingest systemic risk data ───
  ingestSystemicRisk: (data) => {
    // data from GET /api/analytics/systemic-risk
    set({
      advancedMetrics: {
        aggregate_debtrank: data.debt_rank?.aggregate ?? null,
        cascade_depth: data.contagion?.cascade_depth ?? null,
        cascade_potential: data.contagion?.cascade_potential ?? null,
        critical_banks: data.contagion?.critical_banks ?? [],
        systemic_risk_index: data.health?.systemic_risk_index ?? null,
        network_density: data.network?.density ?? null,
        clustering_coefficient: data.network?.clustering_coefficient ?? null,
        avg_path_length: null, // not directly in systemic-risk endpoint
        concentration_index: data.network?.concentration_index ?? null,
        market_price: get().advancedMetrics.market_price,
        interest_rate: get().advancedMetrics.interest_rate,
        market_stress_regime: data.health?.overall_status?.toUpperCase() ?? 'NORMAL',
        liquidity_index: data.health?.liquidity_index ?? null,
        total_losses: get().advancedMetrics.total_losses,
        clearing_iterations: get().advancedMetrics.clearing_iterations,
        avg_recovery_rate: get().advancedMetrics.avg_recovery_rate,
      },
    });

    // Update bank debt ranks from individual scores
    if (data.debt_rank?.individual) {
      set({ bankDebtRanks: data.debt_rank.individual });
    }
  },

  // ─── Ingest DebtRank ranking data ───
  ingestDebtRank: (data) => {
    // data from GET /api/analytics/debtrank
    if (data.individual_ranks || data.rankings) {
      const ranks = {};
      if (data.rankings) {
        data.rankings.forEach((r) => { ranks[String(r.bank_id)] = r.debtrank; });
      } else if (data.individual_ranks) {
        Object.entries(data.individual_ranks).forEach(([k, v]) => { ranks[String(k)] = v; });
      }
      set({ bankDebtRanks: ranks });
    }
  },

  // ─── Ingest market state ───
  ingestMarketState: (data) => {
    // data from GET /api/market/state or market_state in step result
    const mkt = data.asset_market || data;
    set((state) => ({
      advancedMetrics: {
        ...state.advancedMetrics,
        market_price: mkt.price ?? mkt.asset_price ?? state.advancedMetrics.market_price,
        interest_rate: mkt.interest_rate ?? data.lending_market?.interest_rate ?? state.advancedMetrics.interest_rate,
        liquidity_index: data.lending_market?.liquidity ?? mkt.liquidity ?? state.advancedMetrics.liquidity_index,
        market_stress_regime: data.market_regime?.regime ?? state.advancedMetrics.market_stress_regime,
      },
    }));
  },

  // ─── Ingest time-series from backend ───
  ingestTimeSeries: (data) => {
    // data from GET /api/simulation/history/timeseries
    // { metrics, series: { metricName: { timestamps, values } }, count }
    if (!data.series) return;
    const ts = {};
    if (data.series.market_price) ts.market_prices = data.series.market_price.values;
    if (data.series.avg_capital_ratio) ts.system_capital_ratios = data.series.avg_capital_ratio.values;
    if (data.series.default_rate) ts.default_rates = data.series.default_rate.values;
    if (data.series.liquidity_index) ts.liquidity_indices = data.series.liquidity_index.values;
    // interest rate may come as separate metric
    if (data.series.interest_rate) ts.interest_rates = data.series.interest_rate.values;

    set((state) => ({
      timeSeriesHistory: { ...state.timeSeriesHistory, ...ts },
    }));
  },

  // ─── Ingest step result metrics ───
  ingestStepResult: (result) => {
    // result from POST /api/simulation/step
    // { steps_completed, current_step, is_done, rewards, network_stats, market_state, infrastructure }
    const m = result.network_stats || {};
    set({
      metrics: {
        liquidity: m.avg_liquidity ?? m.avg_capital_ratio ?? null,
        default_rate: (m.num_defaulted ?? 0) / Math.max(m.total_banks ?? 1, 1),
        equilibrium_score: m.avg_capital_ratio ? Math.min(1, m.avg_capital_ratio / 0.15) : null,
        volatility: m.volatility ?? null,
      },
    });
  },

  // ─── Actions: State Updates (from WebSocket) ───
  setStateUpdate: (payload) => {
    const { timestep, nodes, edges } = payload;
    const prevHistory = get().bankHistory;
    const newHistory = { ...prevHistory };

    // Accumulate per-bank history
    nodes.forEach((node) => {
      const id = String(node.id);
      const entry = {
        timestep,
        capital_ratio: node.capital_ratio,
        stress: node.stress,
        status: node.status,
      };
      if (!newHistory[id]) {
        newHistory[id] = [];
      }
      newHistory[id] = [...newHistory[id].slice(-(BANK_HISTORY_SIZE - 1)), entry];
    });

    set({
      timestep,
      nodes,
      edges,
      bankHistory: newHistory,
    });
  },

  pushEvent: (event) => {
    set((state) => ({
      events: [...state.events, event],
    }));
  },

  clearEvents: () => set({ events: [] }),

  setMetrics: (payload) => {
    set({ metrics: payload });
  },

  setAdvancedMetrics: (payload) => {
    set({ advancedMetrics: payload });
  },

  pushTimeSeriesData: (data) => {
    set((state) => ({
      timeSeriesHistory: {
        market_prices: [...state.timeSeriesHistory.market_prices, data.market_price].slice(-100),
        interest_rates: [...state.timeSeriesHistory.interest_rates, data.interest_rate].slice(-100),
        liquidity_indices: [...state.timeSeriesHistory.liquidity_indices, data.liquidity_index].slice(-100),
        default_rates: [...state.timeSeriesHistory.default_rates, data.default_rate].slice(-100),
        system_capital_ratios: [...state.timeSeriesHistory.system_capital_ratios, data.system_capital_ratio].slice(-100),
      },
    }));
  },

  setBankDebtRanks: (payload) => {
    set({ bankDebtRanks: payload });
  },

  // ─── Actions: Simulation Control ───
  setSimStatus: (status) => set({ simStatus: status }),

  setScenario: (scenarioId) => set({ selectedScenario: scenarioId }),

  setCustomConfig: (key, value) =>
    set((state) => ({
      customConfig: { ...state.customConfig, [key]: value },
    })),

  // ─── Actions: Bank Selection ───
  toggleBankSelection: (bankId) => {
    const id = String(bankId);
    set((state) => {
      const selected = state.selectedBanks.includes(id)
        ? state.selectedBanks.filter((b) => b !== id)
        : [...state.selectedBanks, id];
      return { selectedBanks: selected };
    });
  },

  selectBank: (bankId) => {
    const id = String(bankId);
    set((state) => {
      if (state.selectedBanks.includes(id)) return {};
      return { selectedBanks: [...state.selectedBanks, id] };
    });
  },

  deselectBank: (bankId) => {
    const id = String(bankId);
    set((state) => ({
      selectedBanks: state.selectedBanks.filter((b) => b !== id),
    }));
  },

  clearBankSelection: () => set({ selectedBanks: [] }),

  toggleBankView: () => set((state) => ({ bankViewOpen: !state.bankViewOpen })),
  setBankViewOpen: (open) => set({ bankViewOpen: open }),

  // ─── Actions: Auth ───
  setAuth: ({ isAuthenticated, currentBankId, currentBankData, restrictedMode }) =>
    set({
      isAuthenticated,
      currentBankId,
      currentBankData,
      restrictedMode,
      showLanding: false, // skip landing after login
    }),

  logout: () =>
    set({
      isAuthenticated: false,
      currentBankId: null,
      currentBankData: null,
      restrictedMode: false,
      showLanding: true,
      apiLoading: false,
      apiError: null,
      backendInitialized: false,
      nodes: [],
      edges: [],
      events: [],
      metrics: {
        liquidity: null,
        default_rate: null,
        equilibrium_score: null,
        volatility: null,
      },
      timestep: 0,
      simStatus: 'idle',
      selectedBanks: [],
      bankHistory: {},
      layoutComputed: false,
      activeView: 'overview',
      nodeDecisions: {},
      advancedMetrics: {
        aggregate_debtrank: null,
        cascade_depth: null,
        cascade_potential: null,
        critical_banks: [],
        systemic_risk_index: null,
        network_density: null,
        clustering_coefficient: null,
        avg_path_length: null,
        concentration_index: null,
        market_price: null,
        interest_rate: null,
        market_stress_regime: 'NORMAL',
        liquidity_index: null,
        total_losses: null,
        clearing_iterations: null,
        avg_recovery_rate: null,
      },
      timeSeriesHistory: {
        market_prices: [],
        interest_rates: [],
        liquidity_indices: [],
        default_rates: [],
        system_capital_ratios: [],
      },
      bankDebtRanks: {},
    }),

  // ─── Actions: View Navigation ───
  setActiveView: (view) => set({ activeView: view }),
  enterApp: () => set({ showLanding: false }),

  // ─── Actions: Node Decisions ───
  setNodeDecision: (nodeId, decision) => set((state) => ({
    nodeDecisions: { ...state.nodeDecisions, [String(nodeId)]: decision },
  })),
  clearNodeDecisions: () => set({ nodeDecisions: {} }),

  // ─── Actions: Layout ───
  setLayoutComputed: (val) => set({ layoutComputed: val }),

  // ─── Actions: Reset ───
  resetAll: () =>
    set({
      apiLoading: false,
      apiError: null,
      backendInitialized: false,
      nodes: [],
      edges: [],
      events: [],
      metrics: {
        liquidity: null,
        default_rate: null,
        equilibrium_score: null,
        volatility: null,
      },
      timestep: 0,
      simStatus: 'idle',
      selectedBanks: [],
      bankHistory: {},
      layoutComputed: false,
      activeView: 'overview',
      nodeDecisions: {},
      advancedMetrics: {
        aggregate_debtrank: null,
        cascade_depth: null,
        cascade_potential: null,
        critical_banks: [],
        systemic_risk_index: null,
        network_density: null,
        clustering_coefficient: null,
        avg_path_length: null,
        concentration_index: null,
        market_price: null,
        interest_rate: null,
        market_stress_regime: 'NORMAL',
        liquidity_index: null,
        total_losses: null,
        clearing_iterations: null,
        avg_recovery_rate: null,
      },
      timeSeriesHistory: {
        market_prices: [],
        interest_rates: [],
        liquidity_indices: [],
        default_rates: [],
        system_capital_ratios: [],
      },
      bankDebtRanks: {},
    }),
}));

export default useSimulationStore;
