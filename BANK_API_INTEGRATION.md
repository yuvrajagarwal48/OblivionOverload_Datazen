# Bank API Integration Documentation

## Overview

This document describes the integration of backend bank APIs from the `ai` branch into the frontend application. The integration adds comprehensive transaction analysis, bank history tracking, and proper What-If simulation capabilities.

## New API Endpoints

### Bank Information APIs

#### 1. List All Banks
```javascript
GET /api/bank/
Response: {
  count: number,
  banks: Array<BankSummary>,
  summary: {
    total_equity: number,
    avg_capital_ratio: number,
    at_risk: number
  }
}
```

**Frontend Function:** `api.listBanks()`

#### 2. Get Bank History
```javascript
GET /api/bank/{bank_id}/history?start=0&end=100
Response: {
  bank_id: number,
  count: number,
  timesteps: number[],
  data: Array<HistoryPoint>,
  series: {
    equity: number[],
    capital_ratio: number[],
    cash: number[],
    status: string[]
  }
}
```

**Frontend Function:** `api.getBankHistory(bankId, start?, end?)`

**Usage in Component:** `TransactionHistory.jsx` uses this to show historical transaction data.

#### 3. Get Bank Transactions
```javascript
GET /api/bank/{bank_id}/transactions?limit=100&tx_type=loan
Response: {
  bank_id: number,
  count: number,
  transactions: Array<Transaction>,
  summary: {
    total_transactions: number,
    total_inflows: number,
    total_outflows: number,
    net_flow: number,
    avg_transaction_size: number,
    counterparty_count: number
  }
}
```

**Frontend Function:** `api.getBankTransactions(bankId, limit?, txType?)`

**Transaction Object:**
```javascript
{
  timestep: number,
  type: string,  // 'loan', 'repayment', 'margin', etc.
  direction: string,  // 'inflow' | 'outflow'
  counterparty: number,  // Bank ID or -1 for system
  amount: number
}
```

#### 4. Get Bank Exposures
```javascript
GET /api/bank/{bank_id}/exposures
Response: {
  bank_id: number,
  assets: {
    count: number,
    total: number,
    details: Array<ExposureDetail>
  },
  liabilities: {
    count: number,
    total: number,
    details: Array<ExposureDetail>
  },
  summary: {
    net_position: number,
    is_net_lender: boolean,
    exposure_to_defaults: number
  }
}
```

**Frontend Function:** `api.getBankExposures(bankId)`

#### 5. Get Stressed Banks
```javascript
GET /api/bank/stressed
Response: {
  stressed: { count: number, banks: Array<BankInfo> },
  critical: { count: number, banks: Array<BankInfo> },
  defaulted: { count: number, banks: Array<BankInfo> },
  summary: {
    total_at_risk: number,
    total_defaulted: number,
    system_health: 'healthy' | 'stressed' | 'crisis'
  }
}
```

**Frontend Function:** `api.getStressedBanks()`

### What-If Analysis APIs

#### 1. Analyze Transaction (Comprehensive)
```javascript
POST /api/whatif/analyze
Body: {
  transaction_type: 'loan_approval' | 'margin_increase' | 'borrow' | 'sell_assets',
  initiator_id: number,
  counterparty_id: number,
  amount: number,
  interest_rate: number,  // default: 0.05
  duration: number,  // default: 10
  collateral: number,  // default: 0
  horizon: number,  // default: 10
  num_simulations: number  // default: 20
}

Response: {
  analysis_id: string,
  timestamp: string,
  transaction: TransactionDetails,
  baseline: {
    initiator_equity: number,
    initiator_capital_ratio: number,
    initiator_survives: boolean,
    counterparty_equity: number,
    counterparty_survives: boolean,
    system_defaults: number,
    total_equity: number
  },
  counterfactual: {
    // Same structure as baseline
  },
  deltas: {
    initiator_equity_change: number,
    initiator_capital_change: number,
    system_defaults_change: number,
    system_equity_change: number
  },
  risk_assessment: {
    initiator_pd: number,
    counterparty_pd: number,
    expected_credit_loss: number,
    loss_given_default: number,
    system_impact: {
      defaults_change: number,
      cascade_probability: number,
      contagion_depth: number
    },
    liquidity_impact: {
      liquidity_drain: number,
      margin_call_probability: number
    },
    overall_risk_score: number,  // 0-100
    risk_category: 'low' | 'moderate' | 'elevated' | 'high' | 'very_high'
  },
  recommendation: {
    decision: 'approve' | 'approve_with_conditions' | 'review_required' | 'reject',
    confidence: number,
    risk_category: string,
    reasons: string[],
    conditions: string[]
  },
  simulation_info: {
    horizon: number,
    num_simulations: number,
    confidence_level: number
  }
}
```

**Frontend Function:** `api.analyzeTransaction(payload)`

**Usage in Component:** `WhatIfPanel.jsx` uses this for comprehensive transaction simulation.

#### 2. Quick Risk Check
```javascript
GET /api/whatif/quick-check/{bank_id}?amount=1000&transaction_type=loan_approval
Response: {
  bank_id: number,
  transaction_type: string,
  amount: number,
  current_state: {
    equity: number,
    capital_ratio: number,
    pd: number
  },
  post_transaction: {
    estimated_capital_ratio: number,
    capital_impact: number
  },
  assessment: {
    risk_level: 'low' | 'moderate' | 'elevated' | 'high',
    recommendation: 'approve' | 'review' | 'reject'
  }
}
```

**Frontend Function:** `api.quickRiskCheck(bankId, amount, transactionType?)`

## Frontend Integration

### 1. WhatIfPanel Component

**File:** `src/components/WhatIfPanel.jsx`

**Changes:**
- Replaced simple action vector API with comprehensive `analyzeTransaction` endpoint
- Maps frontend transaction types (LEND, BORROW, SELL_ASSETS) to backend types
- Parses detailed risk assessment and recommendation from response
- Displays risk score, category, and conditional approval requirements

**API Flow:**
```
User inputs transaction → runSimulation() → api.analyzeTransaction() →
Parse response (baseline, counterfactual, deltas, risk, recommendation) →
Display results with PASS/FAIL badge
```

### 2. TransactionHistory Component

**File:** `src/components/TransactionHistory.jsx`

**Changes:**
- Fetches transaction history from `/api/bank/{id}/transactions` endpoint
- Displays transaction summary (total inflows, outflows, net flow)
- Falls back to activity log if API fails or in mock mode
- Added refresh button to fetch latest transactions on demand
- Supports both API transaction format and activity log format

**Features:**
- **Transaction Summary Card**: Shows total inflows, outflows, and net flow
- **Refresh Button**: Manual refresh of transaction history
- **Dual Format Support**: Handles both backend API and frontend activity log formats
- **Loading States**: Visual feedback during API calls

### 3. API Service Layer

**File:** `src/services/api.js`

**New Functions:**
```javascript
// Bank APIs
listBanks()
getBankHistory(bankId, start, end)
getBankTransactions(bankId, limit, txType)
getBankExposures(bankId)
getStressedBanks()

// What-If APIs
analyzeTransaction(payload)
quickRiskCheck(bankId, amount, transactionType)

// Legacy (still supported)
whatIf(payload)  // Basic action vector endpoint
```

## Backend API Routes Structure

The backend follows a modular router architecture:

```
api/
├── main.py                    # Main FastAPI app with legacy endpoints
├── routes/
│   ├── __init__.py           # Router exports
│   ├── state.py              # Shared simulation state
│   ├── models.py             # Pydantic request/response models
│   ├── simulation.py         # /api/simulation/* routes
│   ├── banks.py              # /api/bank/* routes ⭐
│   ├── whatif.py             # /api/whatif/* routes ⭐
│   ├── analytics.py          # /api/analytics/* routes
│   ├── market.py             # /api/market/* routes
│   ├── infrastructure.py     # /api/infrastructure/* routes
│   └── ai_insights.py        # /api/ai/* routes
```

**Key Files:**
- `banks.py`: Bank details, history, transactions, exposures, stressed banks
- `whatif.py`: What-if analysis, counterfactual simulation, quick risk checks
- `state.py`: Global simulation state management with transaction logging
- `models.py`: Request/response schemas for all endpoints

## Configuration

**Backend Base URL:**
```javascript
// src/config.js
export const API_BASE_URL = 'http://localhost:8000';
export const USE_MOCK = false;  // Set to true for offline development
```

## Mock Mode

When `USE_MOCK = true`, the application uses local mock data:
- Transaction history shows activity log from graph events
- What-If analysis uses simple heuristic calculations
- No backend API calls are made

## Error Handling

All API functions include error handling:
```javascript
try {
  const data = await api.getBankTransactions(bankId, 100);
  // Use data
} catch (err) {
  console.warn('Failed to fetch transactions:', err);
  // Fall back to alternative data source
}
```

## Testing

### Manual Testing Checklist

1. **Start Backend:**
   ```bash
   cd api
   uvicorn main:app --reload
   ```

2. **Test Transaction History:**
   - Login as a bank
   - View Transaction History panel
   - Check summary shows inflows/outflows
   - Click refresh button

3. **Test What-If Simulation:**
   - Login as a bank
   - Go to What-If Simulation panel
   - Select transaction type (LEND/BORROW/SELL_ASSETS)
   - Choose counterparty and amount
   - Click "Run Simulation"
   - Verify risk assessment and recommendation display

4. **Test API Endpoints Directly:**
   ```bash
   # List banks
   curl http://localhost:8000/api/bank/

   # Get bank transactions
   curl http://localhost:8000/api/bank/0/transactions?limit=10

   # What-if analysis
   curl -X POST http://localhost:8000/api/whatif/analyze \
     -H "Content-Type: application/json" \
     -d '{
       "transaction_type": "loan_approval",
       "initiator_id": 0,
       "counterparty_id": 1,
       "amount": 1000,
       "horizon": 10,
       "num_simulations": 20
     }'
   ```

## Future Enhancements

1. **Real-time Updates**: WebSocket support for live transaction streaming
2. **Historical Charts**: Visualize bank metrics over time using history API
3. **Exposure Network Graph**: Interactive visualization of bank exposures
4. **Risk Alerts**: Automated alerts when banks enter stressed/critical states
5. **Batch What-If**: Compare multiple transaction scenarios side-by-side
6. **Export**: Download transaction history and analysis reports

## API Documentation References

- **Main API Docs**: Available at `http://localhost:8000/docs` (Swagger UI)
- **Backend Code**: `ai` branch, `api/routes/` directory
- **Frontend Integration**: `master` branch, `src/services/api.js`

## Support

For issues or questions:
1. Check console logs for API errors
2. Verify backend is running on `http://localhost:8000`
3. Test endpoints directly using Swagger UI
4. Check network tab in browser DevTools
