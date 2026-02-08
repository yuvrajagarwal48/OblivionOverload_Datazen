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

  // ─── Bank Registry (real Indian banks from RBI data) ───
  bankRegistry: [],        // Full list from /api/bank/registry
  registryLoaded: false,
  selectedRealBanks: [],   // Array of bank_id ints selected for simulation
  registrySearchResults: [],

  // ─── Bank Name Map (populated after init from real banks) ───
  bankNameMap: {},         // { [simId]: "State Bank of India" }

  // ─── Manual Configuration (Custom Setup) ───
  manualConfig: {
    banks: [],
    ccps: [],
    market: {
      initial_price: 100,
      interest_rate: 2.5,
      liquidity_index: 0.8,
      volatility: 0.2,
      episode_length: 100,
    },
  },
  useManualSetup: false, // Toggle between scenario and manual setup

  // ─── View State ───
  activeView: 'overview', // 'overview' | 'network' | 'analytics' | 'inspector'

  // ─── Node Decisions (last action per node) ───
  nodeDecisions: {}, // { [nodeId]: 'LEND' | 'HOARD' | 'FIRE_SALE' | ... }

  // ─── Bank History (rolling buffer for time-series) ───
  bankHistory: {}, // { [bankId]: [{ timestep, capital_ratio, stress, status }, ...] }

  // ─── Activity Log (scrollable feed of node + edge activity) ───
  activityLog: [], // [{ id, timestep, type, icon, message, detail, color }]

  // ─── Edge Activity Labels (what is happening on each edge) ───
  edgeActivity: {}, // { 'source-target': { label, type, delta, timestep } }

  // ─── Panel Sizes (for resize functionality) ───
  panelSizes: {
    leftSidebar: 280,
    rightPanel: 300,
    bottomPanel: 220,
  },

  // ─── Layout tracking ───
  layoutComputed: false,

  // ─── Actions: API State ───
  setApiLoading: (val) => set({ apiLoading: val }),
  setApiError: (err) => set({ apiError: err }),
  clearApiError: () => set({ apiError: null }),
  setBackendInitialized: (val) => set({ backendInitialized: val }),

  // ─── Actions: Bank Registry ───
  setBankRegistry: (banks) => set({ bankRegistry: banks, registryLoaded: true }),
  setRegistrySearchResults: (results) => set({ registrySearchResults: results }),
  toggleRealBank: (bankId) => {
    set((state) => {
      const selected = state.selectedRealBanks.includes(bankId)
        ? state.selectedRealBanks.filter((id) => id !== bankId)
        : [...state.selectedRealBanks, bankId];
      return { selectedRealBanks: selected };
    });
  },
  clearRealBankSelection: () => set({ selectedRealBanks: [] }),
  setBankNameMap: (map) => set({ bankNameMap: map }),

  // ─── Helper: generate edges from node list (BA-like topology) ───
  _generateEdgesFromNodes: (nodeList) => {
    if (!nodeList || nodeList.length < 2) return [];
    const edges = [];
    const ids = nodeList.map((n) => String(n.id));
    const edgesPerNode = 2;
    const seen = new Set();
    // Simple deterministic preferential-attachment-like wiring
    for (let i = 1; i < ids.length; i++) {
      const targets = [];
      for (let j = 0; j < Math.min(edgesPerNode, i); j++) {
        // Connect to earlier nodes, biased toward low index (core banks)
        const tIdx = (i + j * 7) % i; // deterministic but spread
        const key = `${ids[i]}-${ids[tIdx]}`;
        const rev = `${ids[tIdx]}-${ids[i]}`;
        if (!seen.has(key) && !seen.has(rev)) {
          seen.add(key);
          targets.push(tIdx);
        }
      }
      targets.forEach((tIdx) => {
        const weight = (nodeList[i].total_assets || nodeList[tIdx].total_assets || 500) * 0.03;
        edges.push({
          source: ids[i],
          target: ids[tIdx],
          weight: Math.round(weight * 100) / 100,
          type: 'credit',
        });
      });
    }
    return edges;
  },

  // ─── Actions: Ingest full metrics snapshot (legacy GET /metrics) ───
  ingestMetrics: (data) => {
    // data from GET /metrics (legacy)
    // { step, network: {num_banks, density, ...}, market: {asset_price, ...}, banks: {[id]: {equity, capital_ratio, cash, status, tier}}, scenario }
    const prevHistory = get().bankHistory;
    const newHistory = { ...prevHistory };
    const step = data.step ?? get().timestep;

    // Convert banks dict → nodes array
    const banks = data.banks || {};
    const nameMap = get().bankNameMap;
    const nodes = Object.entries(banks).map(([bankId, b]) => {
      const id = String(bankId);
      const entry = {
        timestep: step,
        capital_ratio: b.capital_ratio ?? 0,
        stress: 0,
        status: b.status ?? 'active',
      };
      if (!newHistory[id]) newHistory[id] = [];
      newHistory[id] = [...newHistory[id].slice(-49), entry];

      return {
        id,
        label: b.name || nameMap[id] || `Bank ${id}`,
        name: b.name || nameMap[id] || '',
        tier: b.tier ?? 2,
        capital_ratio: b.capital_ratio ?? 0,
        stress: 0,
        status: b.status ?? 'active',
        cash: b.cash ?? 0,
        equity: b.equity ?? 0,
        total_assets: b.total_assets ?? 0,
        total_liabilities: b.total_liabilities ?? 0,
        illiquid_assets: 0,
        external_liabilities: 0,
        interbank_assets: {},
        interbank_liabilities: {},
        debtrank: 0,
        last_updated_timestep: step,
      };
    });

    // Extract network + market metrics
    const net = data.network || {};
    const mkt = data.market || {};

    // Update basic metrics
    const metricsUpdate = {
      liquidity: mkt.liquidity_index ?? net.avg_capital_ratio ?? null,
      default_rate: net.num_banks > 0 ? (net.num_defaulted ?? 0) / net.num_banks : null,
      equilibrium_score: net.avg_capital_ratio ? Math.min(1, net.avg_capital_ratio / 0.15) : null,
      volatility: mkt.volatility ?? null,
    };

    // Push time-series data point
    const prev = get().timeSeriesHistory;
    const tsUpdate = {
      market_prices: [...prev.market_prices, mkt.asset_price ?? null].slice(-100),
      interest_rates: [...prev.interest_rates, mkt.interest_rate ?? null].slice(-100),
      liquidity_indices: [...prev.liquidity_indices, mkt.liquidity_index ?? null].slice(-100),
      default_rates: [...prev.default_rates, metricsUpdate.default_rate].slice(-100),
      system_capital_ratios: [...prev.system_capital_ratios, net.avg_capital_ratio ?? null].slice(-100),
    };

    // ─── Generate edges if store has none ───
    const currentEdges = get().edges;
    let edgesUpdate = {};
    if (currentEdges.length === 0 && nodes.length > 1) {
      const generated = get()._generateEdgesFromNodes(nodes);
      edgesUpdate = { edges: generated, layoutComputed: false };
    }

    // ─── Diff previous nodes to generate activity events ───
    const prevNodes = get().nodes;
    const prevNodeMap = {};
    prevNodes.forEach((n) => { prevNodeMap[String(n.id)] = n; });
    const newLogEntries = [];

    nodes.forEach((n) => {
      const prev = prevNodeMap[n.id];
      if (!prev) return; // first snapshot, no diff

      const bName = n.label || n.name || `Bank ${n.id}`;
      const cashDelta = (n.cash ?? 0) - (prev.cash ?? 0);
      const equityDelta = (n.equity ?? 0) - (prev.equity ?? 0);
      const crDelta = (n.capital_ratio ?? 0) - (prev.capital_ratio ?? 0);

      // Detect DEFAULT
      if (n.status === 'defaulted' && prev.status !== 'defaulted') {
        newLogEntries.push({
          id: `log-${step}-default-${n.id}`,
          timestep: step,
          type: 'DEFAULT',
          icon: '💀',
          message: `${bName} DEFAULTED`,
          detail: `Capital ratio fell to ${(n.capital_ratio * 100).toFixed(1)}%`,
          color: '#ef4444',
          bankId: n.id,
        });
      }
      // Detect LENDING (cash decreased significantly, equity stable or up)
      else if (cashDelta < -5 && equityDelta >= -1) {
        newLogEntries.push({
          id: `log-${step}-lend-${n.id}`,
          timestep: step,
          type: 'LEND',
          icon: '💸',
          message: `${bName} LENT ₹${Math.abs(cashDelta).toFixed(0)}`,
          detail: `Cash: ${prev.cash?.toFixed(0)} → ${n.cash?.toFixed(0)} | CR: ${(n.capital_ratio * 100).toFixed(1)}%`,
          color: '#10b981',
          bankId: n.id,
        });
      }
      // Detect BORROWING (cash increased significantly)
      else if (cashDelta > 5) {
        newLogEntries.push({
          id: `log-${step}-borrow-${n.id}`,
          timestep: step,
          type: 'BORROW',
          icon: '🏦',
          message: `${bName} BORROWED ₹${cashDelta.toFixed(0)}`,
          detail: `Cash: ${prev.cash?.toFixed(0)} → ${n.cash?.toFixed(0)} | CR: ${(n.capital_ratio * 100).toFixed(1)}%`,
          color: '#3b82f6',
          bankId: n.id,
        });
      }
      // Detect FIRE SALE (equity dropped sharply)
      else if (equityDelta < -10 && crDelta < -0.01) {
        newLogEntries.push({
          id: `log-${step}-firesale-${n.id}`,
          timestep: step,
          type: 'FIRE_SALE',
          icon: '🔥',
          message: `${bName} FIRE SALE`,
          detail: `Equity: ${prev.equity?.toFixed(0)} → ${n.equity?.toFixed(0)} (Δ${equityDelta.toFixed(0)})`,
          color: '#f59e0b',
          bankId: n.id,
        });
      }
      // Detect HOARD (cash stable/up, no lending, capital ratio up slightly)
      else if (cashDelta >= 0 && crDelta > 0.005) {
        newLogEntries.push({
          id: `log-${step}-hoard-${n.id}`,
          timestep: step,
          type: 'HOARD',
          icon: '🔒',
          message: `${bName} HOARDING`,
          detail: `CR: ${(prev.capital_ratio * 100).toFixed(1)}% → ${(n.capital_ratio * 100).toFixed(1)}%`,
          color: '#f59e0b',
          bankId: n.id,
        });
      }
      // Detect stress increase
      else if (crDelta < -0.005) {
        newLogEntries.push({
          id: `log-${step}-stress-${n.id}`,
          timestep: step,
          type: 'STRESSED',
          icon: '⚠️',
          message: `${bName} under stress`,
          detail: `CR: ${(prev.capital_ratio * 100).toFixed(1)}% → ${(n.capital_ratio * 100).toFixed(1)}% (Δ${(crDelta * 100).toFixed(1)}%)`,
          color: '#f97316',
          bankId: n.id,
        });
      }
    });

    // Also push synthetic events so the event animations fire
    const syntheticEvents = newLogEntries
      .filter((e) => ['DEFAULT', 'LEND', 'FIRE_SALE', 'HOARD'].includes(e.type))
      .map((e) => ({
        event_id: e.id,
        event_type: e.type,
        from: parseInt(e.bankId),
        timestep: step,
      }));

    // Keep last 200 log entries
    const updatedLog = [...get().activityLog, ...newLogEntries].slice(-200);

    // Update node decisions from derived activity
    const decisions = { ...get().nodeDecisions };
    newLogEntries.forEach((e) => {
      if (e.bankId) decisions[String(e.bankId)] = e.type;
    });

    set({
      timestep: step,
      nodes: nodes.length > 0 ? nodes : get().nodes,
      ...edgesUpdate,
      bankHistory: newHistory,
      metrics: metricsUpdate,
      timeSeriesHistory: tsUpdate,
      activityLog: updatedLog,
      nodeDecisions: decisions,
      events: [...get().events, ...syntheticEvents],
      advancedMetrics: {
        ...get().advancedMetrics,
        network_density: net.density ?? get().advancedMetrics.network_density,
        clustering_coefficient: net.clustering_coefficient ?? get().advancedMetrics.clustering_coefficient,
        market_price: mkt.asset_price ?? get().advancedMetrics.market_price,
        interest_rate: mkt.interest_rate ?? get().advancedMetrics.interest_rate,
        liquidity_index: mkt.liquidity_index ?? get().advancedMetrics.liquidity_index,
      },
    });
  },

  // ─── Ingest network edges (from /network/topology) with diff for activity labels ───
  ingestEdges: (edgeList) => {
    const prevEdges = get().edges;
    const prevEdgeMap = {};
    prevEdges.forEach((e) => {
      prevEdgeMap[`${e.source}-${e.target}`] = e;
    });

    const step = get().timestep;
    const newEdgeActivity = {}; // Fresh state for current edges
    const edgeLogEntries = [];

    const edges = edgeList.map((e) => {
      const src = String(e.source);
      const tgt = String(e.target);
      const key = `${src}-${tgt}`;
      const prevEdge = prevEdgeMap[key];
      const weight = e.weight ?? 0;

      // Always show current weight on edge
      const currentLabel = weight > 1 ? `₹${weight.toFixed(0)}` : '';

      if (prevEdge) {
        const delta = weight - (prevEdge.weight ?? 0);
        if (Math.abs(delta) > 0.5) {
          const type = delta > 0 ? 'LENDING' : 'REPAYMENT';
          const fullMessage = `Bank ${src} ${delta > 0 ? '→' : '←'} Bank ${tgt}: ₹${Math.abs(delta).toFixed(0)} (Total: ₹${weight.toFixed(0)})`;
          newEdgeActivity[key] = { label: currentLabel, message: fullMessage, type, delta, timestep: step };
          edgeLogEntries.push({
            id: `log-${step}-edge-${key}`,
            timestep: step,
            type,
            icon: delta > 0 ? '→' : '←',
            message: `B${src} ${delta > 0 ? '→ lent to' : '← repaid by'} B${tgt}: ₹${Math.abs(delta).toFixed(0)}`,
            detail: `Bank ${src} → Bank ${tgt}`,
            color: delta > 0 ? '#3b82f6' : '#10b981',
          });
        } else {
          // No significant change, just show current weight
          newEdgeActivity[key] = { label: currentLabel, message: `${src} → ${tgt}: ₹${weight.toFixed(0)}`, type: 'STATIC', delta: 0, timestep: step };
        }
      } else {
        // New edge = new lending relationship
        if (weight > 1) {
          const fullMessage = `New Link: Bank ${src} → Bank ${tgt} (₹${weight.toFixed(0)})`;
          newEdgeActivity[key] = { label: currentLabel, message: fullMessage, type: 'NEW_LINK', delta: weight, timestep: step };
          edgeLogEntries.push({
            id: `log-${step}-newedge-${key}`,
            timestep: step,
            type: 'NEW_LINK',
            icon: '🔗',
            message: `New link: B${src} → B${tgt} (₹${weight.toFixed(0)})`,
            detail: `Bank ${src} → Bank ${tgt}`,
            color: '#8b5cf6',
          });
        } else {
          newEdgeActivity[key] = { label: currentLabel, message: `${src} → ${tgt}: ₹${weight.toFixed(0)}`, type: 'STATIC', delta: 0, timestep: step };
        }
      }

      return {
        source: src,
        target: tgt,
        weight,
        type: e.type || 'credit',
      };
    });

    const updatedLog = edgeLogEntries.length > 0
      ? [...get().activityLog, ...edgeLogEntries].slice(-200)
      : get().activityLog;

    set({ edges, edgeActivity: newEdgeActivity, activityLog: updatedLog });
    
    // If topology includes nodes (may include CCP nodes), merge them
    // This will be called separately by useSimulationControl after fetching topology
  },

  // ─── Ingest topology nodes (may include CCPs) ───
  ingestTopologyNodes: (nodeList) => {
    if (!nodeList || nodeList.length === 0) return;
    
    const currentNodes = get().nodes;
    const nodeMap = {};
    currentNodes.forEach(n => { nodeMap[String(n.id)] = n; });
    
    // Merge topology nodes with existing nodes (add CCPs, update banks)
    const mergedNodes = nodeList.map(n => {
      const id = String(n.id);
      const existing = nodeMap[id];
      
      return {
        ...existing,
        id,
        label: n.label || existing?.label || `Node ${id}`,
        tier: n.tier ?? existing?.tier ?? 3,
        capital_ratio: n.capital_ratio ?? existing?.capital_ratio ?? 0,
        status: n.status ?? existing?.status ?? 'active',
        stress: n.stress ?? existing?.stress ?? 0,
        cash: n.cash ?? existing?.cash ?? 0,
        equity: n.equity ?? existing?.equity ?? 0,
        node_type: n.node_type || (id.startsWith('CCP') ? 'ccp' : 'bank'), // Mark CCPs
      };
    });
    
    set({ nodes: mergedNodes });
  },

  // ─── Ingest risk metrics (legacy GET /metrics/risk) ───
  ingestRiskMetrics: (data) => {
    // data from GET /metrics/risk — RiskAnalyzer full report
    // May contain: debt_rank, contagion, network, health, etc.
    const prev = get().advancedMetrics;
    set({
      advancedMetrics: {
        ...prev,
        aggregate_debtrank: data.debt_rank?.aggregate ?? data.aggregate_debtrank ?? prev.aggregate_debtrank,
        cascade_depth: data.contagion?.cascade_depth ?? prev.cascade_depth,
        cascade_potential: data.contagion?.cascade_potential ?? prev.cascade_potential,
        critical_banks: data.contagion?.critical_banks ?? prev.critical_banks,
        systemic_risk_index: data.health?.systemic_risk_index ?? data.systemic_risk_index ?? prev.systemic_risk_index,
        network_density: data.network?.density ?? prev.network_density,
        clustering_coefficient: data.network?.clustering_coefficient ?? prev.clustering_coefficient,
        avg_path_length: data.network?.avg_path_length ?? prev.avg_path_length,
        concentration_index: data.network?.concentration_index ?? prev.concentration_index,
        market_stress_regime: data.health?.overall_status?.toUpperCase() ?? prev.market_stress_regime,
        total_losses: data.clearing?.total_losses ?? prev.total_losses,
        clearing_iterations: data.clearing?.iterations ?? prev.clearing_iterations,
        avg_recovery_rate: data.clearing?.avg_recovery_rate ?? prev.avg_recovery_rate,
      },
    });

    // Update bank debt ranks from individual scores
    if (data.debt_rank?.individual) {
      set({ bankDebtRanks: data.debt_rank.individual });
    } else if (data.bank_risks) {
      const ranks = {};
      Object.entries(data.bank_risks).forEach(([k, v]) => {
        ranks[String(k)] = v.debtrank ?? v.systemic_importance ?? 0;
      });
      set({ bankDebtRanks: ranks });
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

  // (ingestMarketState and ingestTimeSeries removed — data now flows through ingestMetrics)

  // ─── Ingest full state from GET /api/simulation/state ───
  ingestSimulationState: (data) => {
    // data = { timestep, market, network_stats, banks, exchanges, ccps }
    const step = data.timestep ?? get().timestep;
    const nameMap = get().bankNameMap;
    const prevHistory = get().bankHistory;
    const newHistory = { ...prevHistory };

    const banks = data.banks || {};
    const nodes = Object.entries(banks).map(([bankId, b]) => {
      const id = String(bankId);
      const entry = {
        timestep: step,
        capital_ratio: b.capital_ratio ?? 0,
        stress: 0,
        status: b.status ?? 'active',
      };
      if (!newHistory[id]) newHistory[id] = [];
      newHistory[id] = [...newHistory[id].slice(-49), entry];

      return {
        id,
        label: b.name || nameMap[id] || `Bank ${id}`,
        name: b.name || nameMap[id] || '',
        tier: b.tier ?? 2,
        capital_ratio: b.capital_ratio ?? 0,
        stress: 0,
        status: b.status ?? 'active',
        cash: b.cash ?? 0,
        equity: b.equity ?? 0,
        total_assets: b.total_assets ?? 0,
        total_liabilities: b.total_liabilities ?? 0,
        debtrank: 0,
        last_updated_timestep: step,
      };
    });

    const net = data.network_stats || {};
    const mkt = data.market || {};

    // Generate edges if store has none
    const currentEdges = get().edges;
    let edgesUpdate = {};
    if (currentEdges.length === 0 && nodes.length > 1) {
      const generated = get()._generateEdgesFromNodes(nodes);
      edgesUpdate = { edges: generated, layoutComputed: false };
    }

    const metricsUpdate = {
      liquidity: mkt.liquidity_index ?? net.avg_capital_ratio ?? null,
      default_rate: net.num_banks > 0 ? (net.num_defaulted ?? 0) / net.num_banks : null,
      equilibrium_score: net.avg_capital_ratio ? Math.min(1, net.avg_capital_ratio / 0.15) : null,
      volatility: mkt.volatility ?? null,
    };

    const prev = get().timeSeriesHistory;
    const tsUpdate = {
      market_prices: [...prev.market_prices, mkt.asset_price ?? null].slice(-100),
      interest_rates: [...prev.interest_rates, mkt.interest_rate ?? null].slice(-100),
      liquidity_indices: [...prev.liquidity_indices, mkt.liquidity_index ?? null].slice(-100),
      default_rates: [...prev.default_rates, metricsUpdate.default_rate].slice(-100),
      system_capital_ratios: [...prev.system_capital_ratios, net.avg_capital_ratio ?? null].slice(-100),
    };

    set({
      timestep: step,
      nodes: nodes.length > 0 ? nodes : get().nodes,
      ...edgesUpdate,
      bankHistory: newHistory,
      metrics: metricsUpdate,
      timeSeriesHistory: tsUpdate,
      advancedMetrics: {
        ...get().advancedMetrics,
        network_density: net.density ?? get().advancedMetrics.network_density,
        clustering_coefficient: net.clustering_coefficient ?? get().advancedMetrics.clustering_coefficient,
        market_price: mkt.asset_price ?? get().advancedMetrics.market_price,
        interest_rate: mkt.interest_rate ?? get().advancedMetrics.interest_rate,
        liquidity_index: mkt.liquidity_index ?? get().advancedMetrics.liquidity_index,
      },
    });
  },

  // ─── Ingest step result metrics (from POST /api/simulation/step) ───
  ingestStepResult: (result) => {
    // New format: { steps_completed, current_step, is_done, rewards, network_stats, market_state, infrastructure }
    // Legacy format: { step, rewards, done, network_stats, market_state }
    const m = result.network_stats || {};
    const mkt = result.market_state || {};
    const numBanks = m.num_banks || 30;
    const step = result.current_step ?? result.step ?? get().timestep;

    set({
      timestep: step,
      metrics: {
        liquidity: mkt.liquidity_index ?? m.avg_capital_ratio ?? null,
        default_rate: numBanks > 0 ? (m.num_defaulted ?? 0) / numBanks : null,
        equilibrium_score: m.avg_capital_ratio ? Math.min(1, m.avg_capital_ratio / 0.15) : null,
        volatility: mkt.volatility ?? null,
      },
      advancedMetrics: {
        ...get().advancedMetrics,
        network_density: m.density ?? get().advancedMetrics.network_density,
        clustering_coefficient: m.clustering_coefficient ?? get().advancedMetrics.clustering_coefficient,
        market_price: mkt.asset_price ?? get().advancedMetrics.market_price,
        interest_rate: mkt.interest_rate ?? get().advancedMetrics.interest_rate,
        liquidity_index: mkt.liquidity_index ?? get().advancedMetrics.liquidity_index,
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

  // ─── Actions: Manual Configuration ───
  setManualConfig: (key, value) =>
    set((state) => ({
      manualConfig: { ...state.manualConfig, [key]: value },
    })),

  setUseManualSetup: (value) => set({ useManualSetup: value }),

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

  loginAsBank: (bankId) => {
    const state = get();
    const bankNode = state.nodes.find(n => String(n.id) === String(bankId));
    if (bankNode) {
      set({
        currentBankId: String(bankId),
        currentBankData: {
          bank_id: bankNode.id,
          name: bankNode.name || bankNode.label || `Bank ${bankNode.id}`,
          tier: bankNode.tier,
          node_type: bankNode.node_type || 'bank',
        },
        restrictedMode: true,
      });
    }
  },

  switchToObserver: () => set({
    restrictedMode: false,
    currentBankId: null,
    currentBankData: null,
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
      selectedRealBanks: [],
      bankNameMap: {},
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
      activityLog: [],
      edgeActivity: {},
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
  enterApp: () => set({ showLanding: false, isAuthenticated: true }),

  // ─── Actions: Node Decisions ───
  setNodeDecision: (nodeId, decision) => set((state) => ({
    nodeDecisions: { ...state.nodeDecisions, [String(nodeId)]: decision },
  })),
  clearNodeDecisions: () => set({ nodeDecisions: {} }),

  // ─── Actions: Layout ───
  setLayoutComputed: (val) => set({ layoutComputed: val }),
  
  setPanelSize: (panel, size) => 
    set((state) => ({
      panelSizes: { ...state.panelSizes, [panel]: size },
    })),

  // ─── Actions: Reset ───
  resetAll: () =>
    set({
      apiLoading: false,
      apiError: null,
      backendInitialized: false,
      selectedRealBanks: [],
      bankNameMap: {},
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
      activityLog: [],
      edgeActivity: {},
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
