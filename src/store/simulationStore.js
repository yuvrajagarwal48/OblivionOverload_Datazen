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
