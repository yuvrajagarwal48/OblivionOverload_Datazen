# Quick Reference: Bank API Integration

## 🚀 Quick Start

### Backend Setup
```bash
cd api
uvicorn main:app --reload
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Frontend Setup
```bash
npm start
# App available at http://localhost:3000
```

### Configuration
```javascript
// src/config.js
export const API_BASE_URL = 'http://localhost:8000';
export const USE_MOCK = false;  // false = use real API, true = mock mode
```

---

## 📡 New API Endpoints

### Bank Information
```javascript
import * as api from '../services/api';

// Get all banks with risk scores
const banks = await api.listBanks();

// Get transaction history
const history = await api.getBankTransactions(bankId, 100);
// Returns: { transactions, summary: { total_inflows, total_outflows, net_flow } }

// Get historical metrics
const metrics = await api.getBankHistory(bankId, 0, 100);
// Returns: { timesteps, series: { equity, capital_ratio, cash, status } }

// Get exposure breakdown
const exposures = await api.getBankExposures(bankId);
// Returns: { assets, liabilities, summary }

// Get stressed banks
const stressed = await api.getStressedBanks();
// Returns: { stressed, critical, defaulted, summary }
```

### What-If Analysis
```javascript
// Comprehensive analysis (recommended)
const analysis = await api.analyzeTransaction({
  transaction_type: 'loan_approval',  // or 'borrow', 'sell_assets', 'margin_increase'
  initiator_id: 0,
  counterparty_id: 1,
  amount: 1000,
  interest_rate: 0.05,
  duration: 10,
  horizon: 10,
  num_simulations: 20
});
// Returns: { baseline, counterfactual, deltas, risk_assessment, recommendation }

// Quick risk check
const quickCheck = await api.quickRiskCheck(bankId, amount, 'loan_approval');
// Returns: { current_state, post_transaction, assessment }
```

---

## 🎯 Component Usage

### WhatIfPanel
**Location:** `src/components/WhatIfPanel.jsx`

**Features:**
- ✅ Full Monte Carlo what-if analysis
- ✅ Risk assessment with score (0-100)
- ✅ Approval recommendations
- ✅ Conditional approvals
- ✅ Expected credit loss calculation

**Transaction Types:**
- `LEND` → `loan_approval` (backend)
- `BORROW` → `borrow` (backend)
- `SELL_ASSETS` → `sell_assets` (backend)

### TransactionHistory
**Location:** `src/components/TransactionHistory.jsx`

**Features:**
- ✅ Transaction log from backend
- ✅ Summary statistics (inflows, outflows, net)
- ✅ Manual refresh button
- ✅ Fallback to activity log
- ✅ Loading states

---

## 📊 Response Structures

### Transaction Analysis Response
```javascript
{
  analysis_id: "WIF_20260208_...",
  baseline: {
    initiator_capital_ratio: 0.08,
    system_defaults: 2,
    total_equity: 50000
  },
  counterfactual: {
    initiator_capital_ratio: 0.075,
    system_defaults: 2.5,
    total_equity: 49500
  },
  risk_assessment: {
    overall_risk_score: 35.2,
    risk_category: "moderate",
    expected_credit_loss: 45.5,
    system_impact: {
      defaults_change: 0.5,
      cascade_probability: 0.05
    }
  },
  recommendation: {
    decision: "approve_with_conditions",
    confidence: 0.70,
    reasons: ["Low initiator default probability"],
    conditions: ["Enhanced monitoring for 30 days"]
  }
}
```

### Transaction History Response
```javascript
{
  bank_id: 0,
  count: 45,
  transactions: [
    {
      timestep: 15,
      type: "loan",
      direction: "outflow",
      counterparty: 3,
      amount: 250.5
    }
  ],
  summary: {
    total_transactions: 45,
    total_inflows: 1250.75,
    total_outflows: 980.25,
    net_flow: 270.50,
    avg_transaction_size: 50.25,
    counterparty_count: 8
  }
}
```

---

## 🔍 Testing Checklist

### Frontend Testing
- [ ] Login as bank (🔑 button or double-click node)
- [ ] Transaction History panel shows
- [ ] Summary shows inflows/outflows/net
- [ ] Refresh button works
- [ ] What-If panel accessible
- [ ] Select transaction type
- [ ] Choose counterparty and amount
- [ ] Run simulation
- [ ] Risk assessment displays
- [ ] Approve/reject buttons work

### API Testing (cURL)
```bash
# List banks
curl http://localhost:8000/api/bank/

# Get transactions
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

# Quick check
curl "http://localhost:8000/api/whatif/quick-check/0?amount=1000&transaction_type=loan_approval"
```

---

## 🐛 Troubleshooting

### Problem: API calls failing
**Solution:**
1. Check backend is running: `http://localhost:8000/health`
2. Verify `API_BASE_URL` in `src/config.js`
3. Check browser console for errors
4. Test endpoint directly in Swagger UI

### Problem: Transaction history empty
**Solution:**
1. Run simulation for a few steps first
2. Check backend logs for errors
3. Try refresh button
4. Check if `USE_MOCK = true` (uses activity log instead)

### Problem: What-If not working
**Solution:**
1. Ensure counterparty selected
2. Ensure amount > 0
3. Check network tab for API errors
4. Verify backend `/api/whatif/analyze` endpoint responding
5. Check console for parsing errors

### Problem: Mock mode not working
**Solution:**
1. Set `USE_MOCK = true` in `src/config.js`
2. Mock mode uses local data, no backend needed
3. Transaction history shows activity log only
4. What-if uses simple heuristics

---

## 📚 Documentation

- **Full API Docs:** `BANK_API_INTEGRATION.md`
- **Integration Summary:** `INTEGRATION_SUMMARY.md`
- **Backend Swagger UI:** `http://localhost:8000/docs`

---

## 🎨 UI Features

### Transaction History
- **Summary Card**: Shows total inflows, outflows, net flow
- **Refresh Button**: Blue circular button with ↻ icon
- **Transaction Table**: Timestep, type, direction, counterparty, amount
- **Color Coding**: Green (incoming), Red (outgoing)

### What-If Panel
- **Transaction Types**: Lend, Borrow, Sell Assets
- **Risk Display**: Before/After capital ratios with trend arrows
- **Result Badge**: Green PASS or Red FAIL
- **Actions**: Approve & Record, Reject buttons
- **Risk Info**: Score, category, expected loss (in details)

---

## 🚦 Status Indicators

**Risk Categories:**
- 🟢 **Low** (0-20): Approve
- 🟡 **Moderate** (20-40): Approve with conditions
- 🟠 **Elevated** (40-60): Review required
- 🔴 **High** (60-80): Reject
- 🔴 **Very High** (80-100): Reject

**Recommendation Decisions:**
- ✅ `approve` - Transaction safe
- ⚠️ `approve_with_conditions` - Requires monitoring
- ⏸️ `review_required` - Manual review needed
- ❌ `reject` - Too risky

---

## 📝 Notes

- All amounts in **₹ Crores** (1 Cr = 10 million)
- Transaction history limited to 100 most recent
- Monte Carlo uses 20 simulations by default
- Risk scores normalized 0-100 scale
- Backend must be on `ai` branch for full API support
