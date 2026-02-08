// FinSim-MAPPO Frontend Configuration

const isDev = process.env.NODE_ENV === 'development';

// ═══ MOCK MODE ═══
// Set to true to run with dummy data (no backend needed)
// Set to false when your Python backend is running on localhost:8000
export const USE_MOCK = process.env.REACT_APP_USE_MOCK === 'true' || false;

// ═══ SUPABASE ═══
export const SUPABASE_URL = process.env.REACT_APP_SUPABASE_URL || '';
export const SUPABASE_ANON_KEY = process.env.REACT_APP_SUPABASE_ANON_KEY || '';

// API URLs — in dev, CRA proxy forwards to localhost:8000
// Backend routes use /api prefix (set in main.py)
export const API_BASE_URL = isDev ? '' : (process.env.REACT_APP_API_URL || '');

// Polling interval for live API mode (ms)
export const POLL_INTERVAL = 800;

// Animation durations (ms)
export const ANIMATION = {
  LEND_DURATION: 800,
  HOARD_DURATION: 600,
  FIRE_SALE_DURATION: 1000,
  PRICE_DROP_DURATION: 1200,
  DEFAULT_DURATION: 1000,
  CLEANUP_DELAY: 100,
};

// Scenario definitions
export const SCENARIOS = [
  {
    id: 'normal',
    name: 'Baseline',
    subtitle: 'No Shock',
    description: 'Normal market conditions — observe equilibrium behavior.',
    iconName: 'scale',
    color: '#10b981',
  },
  {
    id: 'liquidity_crisis',
    name: 'Liquidity Crisis',
    subtitle: 'Interbank Market Freeze',
    description: 'Severe funding stress with interbank market freeze — watch contagion spread.',
    iconName: 'snowflake',
    color: '#3b82f6',
  },
  {
    id: 'asset_crash',
    name: 'Asset Price Crash',
    subtitle: 'Fire Sale Cascade',
    description: 'Sharp decline in asset prices triggering fire sales across the network.',
    iconName: 'flame',
    color: '#ef4444',
  },
  {
    id: 'systemic',
    name: 'Systemic Crisis',
    subtitle: 'Combined Stress',
    description: 'Combined liquidity and asset price stress causing cascading failures.',
    iconName: 'swords',
    color: '#8b5cf6',
  },
];

// Custom scenario options
export const CUSTOM_OPTIONS = {
  shockType: [
    { value: 'liquidity', label: 'Liquidity Freeze' },
    { value: 'asset', label: 'Asset Price Shock' },
  ],
  severity: [
    { value: 'low', label: 'Low' },
    { value: 'medium', label: 'Medium' },
    { value: 'high', label: 'High' },
  ],
  target: [
    { value: 'tier1', label: 'Tier-1 Bank' },
    { value: 'tier2', label: 'Tier-2 Bank' },
    { value: 'random', label: 'Random' },
  ],
  policy: [
    { value: 'greedy', label: 'Greedy' },
    { value: 'stable', label: 'Stable' },
  ],
};

// Health thresholds for color coding
export const HEALTH_THRESHOLDS = {
  good: 0.7,
  warning: 0.4,
  // below warning = critical
};

// Graph constraints (from spec §12)
export const GRAPH_LIMITS = {
  MAX_NODES: 100,
  MAX_EDGES: 500,
  MAX_EVENTS_PER_SEC: 10,
};

// Bank history buffer size
export const BANK_HISTORY_SIZE = 50;
