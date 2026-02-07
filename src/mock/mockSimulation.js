/**
 * Mock Simulation Engine
 *
 * Generates realistic dummy bank networks and runs fake timesteps
 * so the UI can be tested without a backend.
 *
 * Produces: STATE_UPDATE, EVENT, METRICS_UPDATE messages
 * matching the WebSocket contract in UI_SPECS §9.
 */

// ─── Network Generation ───

function generateNetwork(scenario) {
  const config = SCENARIO_CONFIGS[scenario] || SCENARIO_CONFIGS.baseline;
  const nodes = [];
  const edges = [];

  for (let i = 1; i <= config.numBanks; i++) {
    const tier = i <= config.tier1Count ? 1 : i <= config.tier1Count + config.tier2Count ? 2 : 3;
    const baseCapital = tier === 1 ? 0.12 : 0.10;
    const capital_ratio = baseCapital + Math.random() * 0.08;
    const total_assets = tier === 1 ? 5000 + Math.random() * 3000 : 2000 + Math.random() * 1500;
    const equity = total_assets * capital_ratio;
    const cash = total_assets * (0.15 + Math.random() * 0.10);
    const illiquid_assets = total_assets * (0.5 + Math.random() * 0.2);
    
    nodes.push({
      id: String(i),
      tier,
      capital_ratio,
      stress: 0.05 + Math.random() * 0.15,
      status: 'active',
      last_updated_timestep: 0,
      // Full balance sheet
      cash,
      illiquid_assets,
      total_assets,
      equity,
      external_liabilities: total_assets * (0.3 + Math.random() * 0.2),
      interbank_assets: {},
      interbank_liabilities: {},
      debtrank: 0,
    });
  }

  // Create credit exposure edges (directed)
  const edgeCount = Math.min(config.numEdges, config.numBanks * (config.numBanks - 1) / 2);
  const usedPairs = new Set();

  for (let e = 0; e < edgeCount; e++) {
    let src, tgt;
    let attempts = 0;
    do {
      src = Math.ceil(Math.random() * config.numBanks);
      tgt = Math.ceil(Math.random() * config.numBanks);
      attempts++;
    } while ((src === tgt || usedPairs.has(`${src}-${tgt}`)) && attempts < 100);

    if (attempts >= 100) continue;
    usedPairs.add(`${src}-${tgt}`);

    edges.push({
      source: String(src),
      target: String(tgt),
      weight: 10 + Math.floor(Math.random() * 80),
      type: 'credit',
    });
    
    // Update interbank exposures on nodes
    const srcNode = nodes.find(n => n.id === String(src));
    const tgtNode = nodes.find(n => n.id === String(tgt));
    if (srcNode && tgtNode) {
      const amount = 10 + Math.floor(Math.random() * 80);
      srcNode.interbank_assets[String(tgt)] = (srcNode.interbank_assets[String(tgt)] || 0) + amount;
      tgtNode.interbank_liabilities[String(src)] = (tgtNode.interbank_liabilities[String(src)] || 0) + amount;
    }
  }

  return { nodes, edges };
}

const SCENARIO_CONFIGS = {
  baseline: {
    numBanks: 20,
    tier1Count: 3,
    tier2Count: 7,
    numEdges: 45,
    shockIntensity: 0.02,
    defaultThreshold: 0.02,
    contagionRate: 0.03,
  },
  liquidity_shock: {
    numBanks: 25,
    tier1Count: 4,
    tier2Count: 8,
    numEdges: 55,
    shockIntensity: 0.08,
    defaultThreshold: 0.03,
    contagionRate: 0.06,
    initialShockTargets: [1, 2], // Tier-1 banks
  },
  fire_sale: {
    numBanks: 22,
    tier1Count: 3,
    tier2Count: 7,
    numEdges: 50,
    shockIntensity: 0.10,
    defaultThreshold: 0.03,
    contagionRate: 0.08,
    fireSaleEnabled: true,
  },
  greedy_vs_stable: {
    numBanks: 20,
    tier1Count: 3,
    tier2Count: 7,
    numEdges: 45,
    shockIntensity: 0.05,
    defaultThreshold: 0.03,
    contagionRate: 0.05,
    greedyBanks: [1, 4, 8, 12], // These will hoard
  },
  custom: {
    numBanks: 20,
    tier1Count: 3,
    tier2Count: 7,
    numEdges: 45,
    shockIntensity: 0.06,
    defaultThreshold: 0.03,
    contagionRate: 0.05,
  },
};

// ─── Simulation Step Logic ───

function simulateStep(state, scenario, timestep) {
  const config = SCENARIO_CONFIGS[scenario] || SCENARIO_CONFIGS.baseline;
  const events = [];
  const nodes = state.nodes.map((n) => ({ ...n }));
  const edges = state.edges.map((e) => ({ ...e }));

  // Phase 1: Strategic actions
  nodes.forEach((node) => {
    if (node.status === 'defaulted') return;

    const rand = Math.random();

    // Lending events
    if (rand < 0.25 && node.capital_ratio > 0.08) {
      const targets = edges
        .filter((e) => e.source === node.id)
        .map((e) => e.target);
      if (targets.length > 0) {
        const target = targets[Math.floor(Math.random() * targets.length)];
        const amount = 5 + Math.floor(Math.random() * 30);
        events.push({
          event_id: `evt-${timestep}-lend-${node.id}`,
          event_type: 'LEND',
          from: parseInt(node.id),
          to: parseInt(target),
          amount,
          timestamp: timestep,
        });
        // Lending slightly decreases own capital but reduces stress
        node.capital_ratio = Math.max(0, node.capital_ratio - 0.005);
        node.stress = Math.max(0, node.stress - 0.02);
      }
    }

    // Hoarding (more likely in liquidity_shock & greedy_vs_stable)
    const isGreedy = config.greedyBanks?.includes(parseInt(node.id));
    const hoardChance = isGreedy ? 0.5 : scenario === 'liquidity_shock' ? 0.2 : 0.08;
    if (rand > 1 - hoardChance && node.stress > 0.3) {
      events.push({
        event_id: `evt-${timestep}-hoard-${node.id}`,
        event_type: 'HOARD',
        from: parseInt(node.id),
        timestamp: timestep,
      });
      node.stress += 0.05;
      node.capital_ratio += 0.01; // Hoarding preserves capital
    }
  });

  // Phase 2: Apply scenario shocks
  nodes.forEach((node) => {
    if (node.status === 'defaulted') return;

    // Initial shock targets (liquidity_shock)
    if (timestep <= 3 && config.initialShockTargets?.includes(parseInt(node.id))) {
      node.stress += 0.15;
      node.capital_ratio -= 0.04;
    }

    // Fire sale pressure
    if (config.fireSaleEnabled && timestep > 2 && node.stress > 0.5) {
      events.push({
        event_id: `evt-${timestep}-firesale-${node.id}`,
        event_type: 'FIRE_SALE',
        from: parseInt(node.id),
        timestamp: timestep,
      });
      node.capital_ratio -= 0.03;
      node.stress += 0.08;
    }

    // General market shock
    const shockAmount = config.shockIntensity * (0.5 + Math.random());
    node.stress = Math.min(1, node.stress + shockAmount * 0.3);
    node.capital_ratio = Math.max(0, node.capital_ratio - shockAmount * 0.15);

    // Random recovery for some banks
    if (Math.random() < 0.15 && node.stress > 0.1) {
      node.stress = Math.max(0, node.stress - 0.05);
      node.capital_ratio = Math.min(0.25, node.capital_ratio + 0.01);
    }
  });

  // Phase 3: Contagion from defaulted neighbors
  const defaultedIds = new Set(
    nodes.filter((n) => n.status === 'defaulted').map((n) => n.id)
  );

  edges.forEach((edge) => {
    if (defaultedIds.has(edge.source)) {
      const target = nodes.find((n) => n.id === edge.target);
      if (target && target.status !== 'defaulted') {
        target.stress += config.contagionRate;
        target.capital_ratio -= config.contagionRate * 0.5;
      }
    }
  });

  // Phase 4: Default resolution
  nodes.forEach((node) => {
    if (node.status === 'defaulted') return;

    if (node.capital_ratio <= config.defaultThreshold) {
      node.status = 'defaulted';
      node.stress = 1;
      node.capital_ratio = 0;
      events.push({
        event_id: `evt-${timestep}-default-${node.id}`,
        event_type: 'DEFAULT',
        from: parseInt(node.id),
        timestamp: timestep,
      });
    }
  });

  // Phase 5: Price drop events (periodic global shocks)
  if (timestep % 5 === 0 && scenario !== 'baseline') {
    events.push({
      event_id: `evt-${timestep}-pricedrop`,
      event_type: 'PRICE_DROP',
      timestamp: timestep,
    });
    nodes.forEach((n) => {
      if (n.status !== 'defaulted') {
        n.capital_ratio = Math.max(0, n.capital_ratio - 0.015);
        n.stress = Math.min(1, n.stress + 0.03);
      }
    });
  }

  // Update timestep on nodes
  nodes.forEach((n) => {
    n.last_updated_timestep = timestep;
  });

  // Phase 6: Compute metrics
  const activeNodes = nodes.filter((n) => n.status !== 'defaulted');
  const defaultedCount = nodes.length - activeNodes.length;
  const avgCapital = activeNodes.length > 0
    ? activeNodes.reduce((sum, n) => sum + n.capital_ratio, 0) / activeNodes.length
    : 0;
  const avgStress = activeNodes.length > 0
    ? activeNodes.reduce((sum, n) => sum + n.stress, 0) / activeNodes.length
    : 1;

  const metrics = {
    liquidity: Math.max(0, Math.min(1, avgCapital / 0.15)),
    default_rate: defaultedCount / nodes.length,
    equilibrium_score: Math.max(0, Math.min(1, 1 - avgStress)),
    volatility: Math.max(0, Math.min(1, avgStress * 0.8 + (defaultedCount / nodes.length) * 0.2)),
  };

  // Compute advanced metrics
  const advancedMetrics = computeAdvancedMetrics(nodes, edges, timestep, config);
  
  // Compute DebtRank for each bank
  const debtRanks = computeDebtRanks(nodes, edges);

  return { nodes, edges, events, metrics, advancedMetrics, debtRanks };
}

// ─── Advanced Metrics Computation ───

function computeAdvancedMetrics(nodes, edges, timestep, config) {
  const activeNodes = nodes.filter(n => n.status !== 'defaulted');
  const defaultedCount = nodes.length - activeNodes.length;
  
  // Network metrics
  const network_density = edges.length / (nodes.length * (nodes.length - 1));
  const clustering_coefficient = 0.3 + Math.random() * 0.3; // Simplified
  const avg_path_length = 2.5 + Math.random() * 1.5;
  
  // Concentration (simplified - sum of top 3 exposures / total)
  const totalExposure = edges.reduce((sum, e) => sum + e.weight, 0);
  const topExposures = edges.map(e => e.weight).sort((a, b) => b - a).slice(0, 3);
  const concentration_index = topExposures.reduce((s, v) => s + v, 0) / (totalExposure || 1);
  
  // Market data (evolving over time)
  const market_price = 100 - timestep * 0.5 - Math.random() * 5;
  const interest_rate = 0.05 + Math.sin(timestep * 0.1) * 0.02 + defaultedCount * 0.005;
  const liquidity_index = activeNodes.reduce((sum, n) => sum + (n.cash / n.total_assets), 0) / (activeNodes.length || 1);
  
  // Market stress regime
  let market_stress_regime = 'NORMAL';
  if (defaultedCount > nodes.length * 0.3 || liquidity_index < 0.1) {
    market_stress_regime = 'CRISIS';
  } else if (defaultedCount > nodes.length * 0.1 || liquidity_index < 0.15) {
    market_stress_regime = 'STRESSED';
  }
  
  // Systemic risk
  const cascade_depth = defaultedCount > 0 ? Math.floor(1 + Math.log(defaultedCount + 1) * 2) : 0;
  const cascade_potential = Math.min(1, defaultedCount / (nodes.length * 0.5));
  const systemic_risk_index = (defaultedCount / nodes.length) * 0.4 + (1 - liquidity_index) * 0.3 + cascade_potential * 0.3;
  
  // Critical banks (tier-1 banks with high stress)
  const critical_banks = nodes
    .filter(n => n.tier === 1 && n.stress > 0.6 && n.status !== 'defaulted')
    .map(n => parseInt(n.id));
  
  // Clearing metrics (mock)
  const total_losses = defaultedCount * 500 + Math.random() * 200;
  const clearing_iterations = 3 + Math.floor(Math.random() * 5);
  const avg_recovery_rate = defaultedCount > 0 ? 0.6 + Math.random() * 0.3 : 1.0;
  
  // Aggregate DebtRank (sum of all individual debtranks)
  const aggregate_debtrank = nodes.reduce((sum, n) => sum + (n.debtrank || 0), 0);
  
  return {
    aggregate_debtrank,
    cascade_depth,
    cascade_potential,
    critical_banks,
    systemic_risk_index,
    network_density,
    clustering_coefficient,
    avg_path_length,
    concentration_index,
    market_price,
    interest_rate,
    market_stress_regime,
    liquidity_index,
    total_losses,
    clearing_iterations,
    avg_recovery_rate,
  };
}

function computeDebtRanks(nodes, edges) {
  const debtRanks = {};
  const totalAssets = nodes.reduce((sum, n) => sum + n.total_assets, 0);
  
  nodes.forEach(node => {
    // Simplified DebtRank: (total interbank assets / system assets) * stress multiplier
    const ibAssets = Object.values(node.interbank_assets || {}).reduce((s, v) => s + v, 0);
    const baseRank = ibAssets / (totalAssets || 1);
    const stressMultiplier = 1 + node.stress;
    const tierMultiplier = node.tier === 1 ? 1.5 : 1.0;
    
    const debtrank = baseRank * stressMultiplier * tierMultiplier;
    debtRanks[node.id] = Math.min(1, debtrank * 10); // Scale to 0-1
    node.debtrank = debtRanks[node.id];
  });
  
  return debtRanks;
}

// ─── Mock Simulation Runner ───

export class MockSimulation {
  constructor() {
    this.state = null;
    this.scenario = null;
    this.timestep = 0;
    this.intervalId = null;
    this.onStateUpdate = null;
    this.onEvent = null;
    this.onMetricsUpdate = null;
    this.onAdvancedMetricsUpdate = null;
    this.onTimeSeriesUpdate = null;
    this.onDebtRankUpdate = null;
    this.onComplete = null;
    this.maxSteps = 60;
    this.stepDelay = 800; // ms between steps
  }

  /**
   * Start or restart the simulation with a scenario.
   */
  start(scenario, customConfig = null) {
    this.stop();

    this.scenario = scenario;
    this.timestep = 0;

    // Apply custom config overrides
    if (scenario === 'custom' && customConfig) {
      const base = { ...SCENARIO_CONFIGS.custom };
      if (customConfig.severity === 'high') {
        base.shockIntensity = 0.12;
        base.contagionRate = 0.10;
      } else if (customConfig.severity === 'low') {
        base.shockIntensity = 0.02;
        base.contagionRate = 0.02;
      }
      if (customConfig.shockType === 'asset') {
        base.fireSaleEnabled = true;
      }
      if (customConfig.target === 'tier1') {
        base.initialShockTargets = [1, 2];
      } else if (customConfig.target === 'tier2') {
        base.initialShockTargets = [4, 5, 6];
      }
      if (customConfig.policy === 'greedy') {
        base.greedyBanks = [1, 3, 5, 7, 9];
      }
      SCENARIO_CONFIGS._custom_active = base;
      this.scenario = '_custom_active';
    }

    // Generate initial network
    const scenarioKey = this.scenario === '_custom_active' ? '_custom_active' : scenario;
    const configForGen = SCENARIO_CONFIGS[scenarioKey] || SCENARIO_CONFIGS[scenario] || SCENARIO_CONFIGS.baseline;

    // Temporarily set the config for generation
    if (this.scenario === '_custom_active') {
      SCENARIO_CONFIGS._custom_active = configForGen;
    }

    const network = generateNetwork(scenario === 'custom' ? '_custom_active' : scenario);
    this.state = network;
    this.timestep = 1;

    // Emit initial state
    this._emitState();
    this._emitMetrics({
      liquidity: 0.85 + Math.random() * 0.1,
      default_rate: 0,
      equilibrium_score: 0.9 + Math.random() * 0.05,
      volatility: 0.05 + Math.random() * 0.05,
    });
    
    // Emit initial advanced metrics
    const initialAdvanced = computeAdvancedMetrics(this.state.nodes, this.state.edges, 0, configForGen);
    this._emitAdvancedMetrics(initialAdvanced);
    
    // Emit initial time series
    this._emitTimeSeries({
      market_price: initialAdvanced.market_price,
      interest_rate: initialAdvanced.interest_rate,
      liquidity_index: initialAdvanced.liquidity_index,
      default_rate: 0,
      system_capital_ratio: 0.12,
    });
    
    // Emit initial debt ranks
    const initialDebtRanks = computeDebtRanks(this.state.nodes, this.state.edges);
    this._emitDebtRanks(initialDebtRanks);

    // Start stepping
    this.intervalId = setInterval(() => {
      this._step();
    }, this.stepDelay);
  }

  _step() {
    if (!this.state) return;

    this.timestep++;

    const scenarioKey = this.scenario === '_custom_active' ? '_custom_active' : this.scenario;
    const result = simulateStep(this.state, scenarioKey, this.timestep);
    this.state = { nodes: result.nodes, edges: result.edges };

    // Emit state update
    this._emitState();

    // Emit events (staggered slightly for visual effect)
    result.events.forEach((evt, i) => {
      setTimeout(() => {
        this.onEvent?.(evt);
      }, i * 100);
    });

    // Emit metrics
    this._emitMetrics(result.metrics);
    
    // Emit advanced metrics
    this._emitAdvancedMetrics(result.advancedMetrics);
    
    // Emit time series
    const avgCapital = result.nodes.filter(n => n.status !== 'defaulted')
      .reduce((sum, n) => sum + n.capital_ratio, 0) / (result.nodes.length || 1);
    this._emitTimeSeries({
      market_price: result.advancedMetrics.market_price,
      interest_rate: result.advancedMetrics.interest_rate,
      liquidity_index: result.advancedMetrics.liquidity_index,
      default_rate: result.metrics.default_rate,
      system_capital_ratio: avgCapital,
    });
    
    // Emit debt ranks
    this._emitDebtRanks(result.debtRanks);

    // Check completion
    const allDefaulted = result.nodes.every((n) => n.status === 'defaulted');
    if (this.timestep >= this.maxSteps || allDefaulted) {
      this.stop();
      this.onComplete?.();
    }
  }

  _emitState() {
    this.onStateUpdate?.({
      timestep: this.timestep,
      nodes: this.state.nodes,
      edges: this.state.edges,
    });
  }

  _emitMetrics(metrics) {
    this.onMetricsUpdate?.(metrics);
  }
  
  _emitAdvancedMetrics(advancedMetrics) {
    this.onAdvancedMetricsUpdate?.(advancedMetrics);
  }
  
  _emitTimeSeries(data) {
    this.onTimeSeriesUpdate?.(data);
  }
  
  _emitDebtRanks(debtRanks) {
    this.onDebtRankUpdate?.(debtRanks);
  }

  pause() {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }
  }

  resume() {
    if (!this.intervalId && this.state) {
      this.intervalId = setInterval(() => {
        this._step();
      }, this.stepDelay);
    }
  }

  stepOnce() {
    if (this.state) {
      this._step();
    }
  }

  stop() {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }
  }

  reset() {
    this.stop();
    this.state = null;
    this.timestep = 0;
    this.scenario = null;
  }
}

// Singleton instance
let mockInstance = null;

export function getMockSimulation() {
  if (!mockInstance) {
    mockInstance = new MockSimulation();
  }
  return mockInstance;
}
