# Manual Setup Mode

## Overview
Manual Setup mode allows you to configure banks, CCPs, and market parameters manually instead of using predefined scenarios.

## How to Access

1. **Open Configuration Panel** (left sidebar)
2. **Click "Manual Setup" button** (top right)
3. Configure your custom financial network

## Tabs

### 1. Banks Tab
Configure individual banks with custom parameters:

- **Name**: Custom bank identifier
- **Capital**: Initial capital reserves (₹)
- **Assets**: Total asset value (₹)
- **Tier**: Bank tier (1, 2, or 3) - determines importance in network

**Actions:**
- Click "Add Bank" to create a new bank
- Click trash icon to remove a bank
- Modify any field while simulation is idle

### 2. CCPs Tab
Configure Central Counterparties (clearing houses):

- **Name**: CCP identifier
- **Capital**: CCP capital reserves (₹)
- **Clearing Capacity**: Maximum transaction volume (₹)

**Actions:**
- Click "Add CCP" to create a new CCP
- Click trash icon to remove a CCP
- CCPs will appear as purple diamonds in the network graph

### 3. Market Tab
Configure global market parameters:

- **Initial Asset Price**: Starting price for tradeable assets (default: 100)
- **Interest Rate (%)**: Base lending rate for the system (default: 2.5%)
- **Liquidity Index**: Market liquidity measure, 0-1 (default: 0.8)
- **Volatility**: Price volatility factor, 0-1 (default: 0.2)
- **Episode Length**: Number of timesteps to simulate (default: 100)

## Backend Integration

When you initialize a simulation with manual setup, the frontend sends:

```javascript
{
  num_banks: <number of banks>,
  episode_length: <timesteps>,
  scenario: 'custom',
  banks: [
    { id, name, capital, assets, tier },
    ...
  ],
  ccps: [
    { id, name, capital, clearing_capacity },
    ...
  ],
  market: {
    initial_price,
    interest_rate,
    liquidity_index,
    volatility,
    episode_length
  }
}
```

**Backend Requirements:**
The backend should accept this configuration in the `/simulation/init` endpoint and create the appropriate agents and market conditions.

## Switching Modes

- **To Manual Setup**: Click "Manual Setup" button in scenario selector
- **From Manual Setup**: Click "Switch to Scenarios" button at the top

**Note:** Mode switching is disabled while simulation is running. Stop the simulation first.

## Example Configurations

### Small Network (3 banks, 1 CCP)
- Bank 1: Capital ₹150, Assets ₹200, Tier 1
- Bank 2: Capital ₹100, Assets ₹150, Tier 2
- Bank 3: Capital ₹80, Assets ₹120, Tier 2
- CCP 1: Capital ₹50, Capacity ₹300

### Large Network (10 banks, 2 CCPs)
- 3 Tier-1 banks (high capital)
- 5 Tier-2 banks (medium capital)
- 2 Tier-3 banks (low capital)
- 2 CCPs with different capacities

### Stress Test Network
- Configure banks with varying capital ratios
- Set high market volatility (0.7-0.9)
- Low liquidity index (0.2-0.4)
- Higher interest rates (5-10%)

## State Management

Manual configuration is stored in the Zustand store:

```javascript
manualConfig: {
  banks: [],
  ccps: [],
  market: { ... }
}
useManualSetup: boolean
```

Actions:
- `setManualConfig(key, value)` - Update config
- `setUseManualSetup(boolean)` - Toggle mode

## UI Features

- **Real-time validation**: Fields have min/max constraints
- **Empty state**: Shows helpful messages when no entities are configured
- **Auto-generated IDs**: Each entity gets a unique timestamp-based ID
- **Disabled during simulation**: All controls are locked while running

## Notes

- Minimum 1 bank required to run simulation
- CCPs are optional (system works without them)
- All monetary values are in rupees (₹)
- Tier 1 banks are typically the largest and most interconnected
- Market parameters affect all agents globally
