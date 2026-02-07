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
    }),
}));

export default useSimulationStore;
