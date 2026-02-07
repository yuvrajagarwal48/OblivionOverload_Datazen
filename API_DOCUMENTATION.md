# FinSim-MAPPO Backend API Documentation

**Base URL:** `http://localhost:8000/api/v1` (adjust for production)

---

## 📊 Simulation Endpoints

### `GET /simulation/status`
**Description:** Get current simulation state (timestep, metrics, network snapshot)

**Response Structure:**
```json
{
  "timestep": 42,
  "is_running": true,
  "scenario": "baseline",
  "metrics": {
    "liquidity": 0.75,
    "default_rate": 0.03,
    "equilibrium_score": 0.92,
    "volatility": 0.18,
    "system_capital_ratio": 0.35
  },
  "advanced_metrics": {
    "aggregate_debtrank": 12.45,
    "cascade_depth": 3,
    "cascade_potential": 0.67,
    "critical_banks": 2,
    "systemic_risk_index": 0.42,
    "network_density": 0.23,
    "clustering_coefficient": 0.31,
    "avg_path_length": 2.8,
    "concentration_index": 0.56,
    "market_price": 98.5,
    "interest_rate": 0.0325,
    "market_stress_regime": "NORMAL",
    "liquidity_index": 0.78,
    "total_losses": 125000,
    "clearing_iterations": 4,
    "avg_recovery_rate": 0.85
  },
  "network": {
    "nodes": [
      {
        "id": 0,
        "name": "Core Bank 0",
        "tier": 1,
        "capital_ratio": 0.42,
        "stress": 0.15,
        "status": "SOLVENT",
        "cash": 500000,
        "illiquid_assets": 1200000,
        "total_assets": 2500000,
        "equity": 800000,
        "external_liabilities": 600000,
        "interbank_assets": {"1": 50000, "3": 30000},
        "interbank_liabilities": {"2": 40000},
        "debtrank": 0.85
      }
      // ... 29 more banks
    ],
    "edges": [
      {
        "source": 0,
        "target": 1,
        "amount": 50000,
        "type": "INTERBANK_LOAN"
      }
      // ... more edges
    ]
  },
  "events": [
    {
      "timestep": 42,
      "type": "DEFAULT",
      "bank_id": 15,
      "description": "Peripheral Bank 15 defaulted due to liquidity crisis"
    },
    {
      "timestep": 41,
      "type": "FIRE_SALE",
      "bank_id": 8,
      "description": "Peripheral Bank 8 sold $50k assets at 12% discount"
    }
  ]
}
```

---

### `GET /simulation/history?start=0&end=100`
**Description:** Get time-series data for all metrics over specified timestep range

**Query Parameters:**
- `start` (int, default=0): Starting timestep
- `end` (int, optional): Ending timestep (if omitted, returns to current timestep)

**Response Structure:**
```json
{
  "timesteps": [0, 1, 2, 3, ..., 100],
  "time_series": {
    "liquidity": [0.82, 0.81, 0.79, ..., 0.75],
    "default_rate": [0.0, 0.0, 0.03, ..., 0.03],
    "equilibrium_score": [0.95, 0.94, 0.93, ..., 0.92],
    "volatility": [0.10, 0.12, 0.15, ..., 0.18],
    "system_capital_ratio": [0.40, 0.39, 0.37, ..., 0.35],
    "market_prices": [100.0, 99.8, 99.2, ..., 98.5],
    "interest_rates": [0.03, 0.0305, 0.031, ..., 0.0325],
    "liquidity_indices": [0.85, 0.84, 0.82, ..., 0.78],
    "systemic_risk_indices": [0.20, 0.25, 0.32, ..., 0.42]
  }
}
```

---

### `POST /simulation/run`
**Description:** Start a new simulation with specified scenario and parameters

**Request Body:**
```json
{
  "scenario": "baseline",
  "num_steps": 100,
  "config": {
    "num_banks": 30,
    "initial_capital_ratio": 0.40,
    "shock_type": "LIQUIDITY_SHOCK",
    "shock_magnitude": 0.20
  }
}
```

**Response:**
```json
{
  "status": "STARTED",
  "simulation_id": "sim_20260207_1430",
  "timestep": 0,
  "message": "Simulation started with baseline scenario"
}
```

---

### `POST /simulation/pause`
**Description:** Pause the running simulation

**Response:**
```json
{
  "status": "PAUSED",
  "timestep": 42,
  "message": "Simulation paused at timestep 42"
}
```

---

### `POST /simulation/resume`
**Description:** Resume a paused simulation

**Response:**
```json
{
  "status": "RUNNING",
  "timestep": 42,
  "message": "Simulation resumed from timestep 42"
}
```

---

### `POST /simulation/step`
**Description:** Execute one simulation step (when paused)

**Response:**
```json
{
  "status": "PAUSED",
  "timestep": 43,
  "message": "Advanced to timestep 43"
}
```

---

### `POST /simulation/reset`
**Description:** Reset simulation to initial state

**Response:**
```json
{
  "status": "RESET",
  "timestep": 0,
  "message": "Simulation reset to timestep 0"
}
```

---

## 🏦 Bank Endpoints

### `GET /bank/{id}`
**Description:** Get detailed information about a specific bank

**Path Parameters:**
- `id` (int): Bank ID (0-29)

**Response Structure:**
```json
{
  "id": 5,
  "name": "Core Bank 5",
  "tier": 1,
  "capital_ratio": 0.42,
  "stress": 0.15,
  "status": "SOLVENT",
  "balance_sheet": {
    "cash": 500000,
    "illiquid_assets": 1200000,
    "total_assets": 2500000,
    "equity": 800000,
    "external_liabilities": 600000,
    "interbank_assets": 80000,
    "interbank_liabilities": 120000
  },
  "interbank_exposures": {
    "assets": {
      "1": 50000,
      "3": 30000
    },
    "liabilities": {
      "0": 70000,
      "2": 50000
    }
  },
  "risk_metrics": {
    "debtrank": 0.85,
    "systemic_importance": "HIGH",
    "var_95": 125000,
    "cvar_95": 180000
  },
  "recent_actions": [
    {
      "timestep": 41,
      "action": "LEND",
      "counterparty": 12,
      "amount": 25000
    },
    {
      "timestep": 40,
      "action": "HOLD",
      "counterparty": null,
      "amount": 0
    }
  ]
}
```

---

### `GET /bank/{id}/history`
**Description:** Get full transaction history for a specific bank

**Path Parameters:**
- `id` (int): Bank ID (0-29)

**Query Parameters:**
- `start_timestep` (int, default=0): Starting timestep
- `end_timestep` (int, optional): Ending timestep
- `action_type` (string, optional): Filter by action type (LEND, BORROW, SELL_ASSETS, etc.)

**Response Structure:**
```json
{
  "bank_id": 5,
  "total_transactions": 42,
  "transactions": [
    {
      "timestep": 42,
      "action": "LEND",
      "counterparty_id": 12,
      "counterparty_name": "Peripheral Bank 12",
      "amount": 25000,
      "outcome": "SUCCESS",
      "capital_ratio_before": 0.41,
      "capital_ratio_after": 0.42,
      "stress_before": 0.18,
      "stress_after": 0.15
    },
    {
      "timestep": 41,
      "action": "HOLD",
      "counterparty_id": null,
      "counterparty_name": null,
      "amount": 0,
      "outcome": "SUCCESS",
      "capital_ratio_before": 0.40,
      "capital_ratio_after": 0.41,
      "stress_before": 0.20,
      "stress_after": 0.18
    }
    // ... more transactions
  ]
}
```

---

## 🔮 What-If Analysis

### `POST /simulation/whatif`
**Description:** Run counterfactual analysis on a proposed transaction

**Request Body:**
```json
{
  "initiator_bank_id": 5,
  "action": "LEND",
  "counterparty_id": 12,
  "amount": 50000,
  "context": {
    "current_timestep": 42,
    "lookahead_steps": 10
  }
}
```

**Response Structure:**
```json
{
  "transaction": {
    "initiator_bank_id": 5,
    "action": "LEND",
    "counterparty_id": 12,
    "amount": 50000
  },
  "evaluation": {
    "outcome": "PASS",
    "confidence": 0.87,
    "risk_assessment": "LOW"
  },
  "impact": {
    "initiator": {
      "capital_ratio_before": 0.42,
      "capital_ratio_after": 0.40,
      "stress_before": 0.15,
      "stress_after": 0.22,
      "debtrank_before": 0.85,
      "debtrank_after": 0.88
    },
    "counterparty": {
      "capital_ratio_before": 0.28,
      "capital_ratio_after": 0.32,
      "stress_before": 0.45,
      "stress_after": 0.38,
      "debtrank_before": 0.15,
      "debtrank_after": 0.18
    },
    "system": {
      "default_rate_before": 0.03,
      "default_rate_after": 0.03,
      "systemic_risk_before": 0.42,
      "systemic_risk_after": 0.40,
      "cascade_depth_before": 3,
      "cascade_depth_after": 2
    }
  },
  "lookahead": {
    "trajectories": [
      {
        "timestep": 43,
        "initiator_capital_ratio": 0.40,
        "counterparty_capital_ratio": 0.32,
        "default_probability": 0.02
      },
      {
        "timestep": 44,
        "initiator_capital_ratio": 0.41,
        "counterparty_capital_ratio": 0.33,
        "default_probability": 0.01
      }
      // ... up to lookahead_steps
    ]
  },
  "explanation": {
    "reasoning": "Transaction reduces systemic risk by improving counterparty liquidity position. Initiator remains well-capitalized with low default probability.",
    "key_factors": [
      "Counterparty stress reduced from 0.45 to 0.38",
      "System cascade depth reduced from 3 to 2",
      "No critical banks in default path"
    ],
    "recommendation": "APPROVE"
  }
}
```

---

## 📈 Analytics Endpoints

### `GET /analytics/systemic-risk`
**Description:** Get comprehensive systemic risk dashboard metrics

**Response Structure:**
```json
{
  "timestamp": "2026-02-07T14:30:00Z",
  "timestep": 42,
  "risk_overview": {
    "systemic_risk_index": 0.42,
    "aggregate_debtrank": 12.45,
    "cascade_depth": 3,
    "cascade_potential": 0.67,
    "critical_banks_count": 2,
    "default_probability": 0.15
  },
  "critical_banks": [
    {
      "bank_id": 15,
      "name": "Peripheral Bank 15",
      "tier": 2,
      "stress": 0.82,
      "capital_ratio": 0.08,
      "debtrank": 0.45,
      "default_probability": 0.75
    },
    {
      "bank_id": 8,
      "name": "Peripheral Bank 8",
      "tier": 2,
      "stress": 0.68,
      "capital_ratio": 0.12,
      "debtrank": 0.38,
      "default_probability": 0.55
    }
  ],
  "cascade_analysis": {
    "initial_shocks": [15],
    "first_round_defaults": [8, 22],
    "second_round_defaults": [18],
    "total_affected": 4,
    "total_losses": 450000,
    "recovery_rate": 0.72
  },
  "stress_distribution": {
    "low": 18,
    "medium": 8,
    "high": 4
  }
}
```

---

### `GET /analytics/network`
**Description:** Get network topology and connectivity metrics

**Response Structure:**
```json
{
  "timestamp": "2026-02-07T14:30:00Z",
  "timestep": 42,
  "topology": {
    "num_nodes": 30,
    "num_edges": 85,
    "network_density": 0.23,
    "clustering_coefficient": 0.31,
    "avg_path_length": 2.8,
    "connected_components": 1,
    "is_strongly_connected": true
  },
  "centrality": {
    "degree_centrality": [
      {"bank_id": 0, "score": 0.52},
      {"bank_id": 1, "score": 0.48},
      {"bank_id": 2, "score": 0.45}
      // ... top 10
    ],
    "betweenness_centrality": [
      {"bank_id": 2, "score": 0.38},
      {"bank_id": 0, "score": 0.35},
      {"bank_id": 3, "score": 0.32}
      // ... top 10
    ],
    "eigenvector_centrality": [
      {"bank_id": 1, "score": 0.42},
      {"bank_id": 0, "score": 0.40},
      {"bank_id": 4, "score": 0.38}
      // ... top 10
    ]
  },
  "communities": [
    {"community_id": 0, "banks": [0, 1, 2, 3, 4, 5], "size": 6},
    {"community_id": 1, "banks": [6, 7, 8, 9, 10], "size": 5}
    // ... more communities
  ],
  "concentration": {
    "herfindahl_index": 0.56,
    "gini_coefficient": 0.42,
    "top_5_share": 0.68
  }
}
```

---

### `GET /analytics/debtrank`
**Description:** Get systemic importance rankings using DebtRank algorithm

**Response Structure:**
```json
{
  "timestamp": "2026-02-07T14:30:00Z",
  "timestep": 42,
  "aggregate_debtrank": 12.45,
  "rankings": [
    {
      "rank": 1,
      "bank_id": 0,
      "name": "Core Bank 0",
      "tier": 1,
      "debtrank": 0.85,
      "systemic_importance": "CRITICAL",
      "stress": 0.15,
      "total_assets": 2500000,
      "interbank_assets": 80000,
      "impact_if_default": 0.32
    },
    {
      "rank": 2,
      "bank_id": 1,
      "name": "Core Bank 1",
      "tier": 1,
      "debtrank": 0.82,
      "systemic_importance": "CRITICAL",
      "stress": 0.18,
      "total_assets": 2400000,
      "interbank_assets": 75000,
      "impact_if_default": 0.30
    }
    // ... all 30 banks ranked
  ],
  "tier_breakdown": {
    "tier_1": {
      "avg_debtrank": 0.78,
      "total_impact": 1.85
    },
    "tier_2": {
      "avg_debtrank": 0.35,
      "total_impact": 0.88
    }
  }
}
```

---

## 🏪 Market Endpoints

### `GET /market/state`
**Description:** Get current market conditions and asset prices

**Response Structure:**
```json
{
  "timestamp": "2026-02-07T14:30:00Z",
  "timestep": 42,
  "asset_market": {
    "price": 98.5,
    "volume": 125000,
    "bid_ask_spread": 0.25,
    "price_change_24h": -1.5,
    "volatility": 0.18
  },
  "lending_market": {
    "interest_rate": 0.0325,
    "total_lending": 850000,
    "total_borrowing": 800000,
    "rate_change_24h": 0.0025,
    "liquidity": 0.78
  },
  "market_regime": {
    "regime": "NORMAL",
    "stress_indicator": 0.25,
    "regime_confidence": 0.88,
    "time_in_regime": 15
  },
  "fire_sales": {
    "active_sales": 2,
    "total_volume": 75000,
    "avg_discount": 0.12,
    "price_impact": -0.8
  }
}
```

---

## 🏛️ CCP (Central Counterparty) Endpoints

### `GET /ccp/status`
**Description:** Get clearing house operational status and guarantees

**Response Structure:**
```json
{
  "timestamp": "2026-02-07T14:30:00Z",
  "timestep": 42,
  "operational": {
    "is_active": true,
    "collateral_pool": 1500000,
    "margin_requirements": 1200000,
    "default_fund": 500000,
    "utilization_ratio": 0.35
  },
  "clearing_summary": {
    "total_cleared": 2500000,
    "num_transactions": 85,
    "failed_transactions": 2,
    "avg_clearing_time": 150,
    "last_clearing_timestep": 42
  },
  "exposures": [
    {
      "bank_id": 0,
      "name": "Core Bank 0",
      "gross_exposure": 180000,
      "net_exposure": 80000,
      "collateral_posted": 90000,
      "margin_shortfall": 0
    }
    // ... all clearing members
  ],
  "risk_metrics": {
    "aggregate_exposure": 2400000,
    "var_99": 320000,
    "stress_loss": 450000,
    "cover_ratio": 3.3
  }
}
```

---

## 🛠️ Utility Endpoints

### `GET /health`
**Description:** Health check endpoint

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-02-07T14:30:00Z",
  "version": "1.0.0"
}
```

---

### `GET /config`
**Description:** Get current simulation configuration

**Response:**
```json
{
  "num_banks": 30,
  "num_tiers": 2,
  "initial_capital_ratio": 0.40,
  "shock_type": "LIQUIDITY_SHOCK",
  "shock_magnitude": 0.20,
  "clearing_method": "EISENBERG_NOE",
  "ccp_enabled": true,
  "belief_learning_enabled": true
}
```

---

## 🔐 Authentication (Supabase Integration)

All endpoints require authentication via Supabase JWT token in `Authorization` header:

```
Authorization: Bearer <supabase_jwt_token>
```

Banks can only access their own data (`/bank/{id}` endpoints) unless they have admin privileges.

---

## 📝 Error Responses

All endpoints return standard error format:

```json
{
  "error": {
    "code": "BANK_NOT_FOUND",
    "message": "Bank with ID 99 does not exist",
    "details": {
      "requested_id": 99,
      "valid_range": [0, 29]
    }
  }
}
```

**Common Error Codes:**
- `BANK_NOT_FOUND` (404)
- `SIMULATION_NOT_RUNNING` (409)
- `INVALID_TRANSACTION` (400)
- `UNAUTHORIZED` (401)
- `FORBIDDEN` (403)
- `INTERNAL_ERROR` (500)

---

## 🚀 Usage Examples

### Frontend Integration Pattern

```javascript
// src/services/api.js
const BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1';

export const api = {
  async getSimulationStatus() {
    const res = await fetch(`${BASE_URL}/simulation/status`);
    return res.json();
  },
  
  async getBankDetails(bankId) {
    const res = await fetch(`${BASE_URL}/bank/${bankId}`);
    return res.json();
  },
  
  async runWhatIf(txData) {
    const res = await fetch(`${BASE_URL}/simulation/whatif`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(txData)
    });
    return res.json();
  }
};
```

---

**Last Updated:** February 7, 2026  
**Version:** 1.0.0  
**Contact:** Backend team for schema changes
