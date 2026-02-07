# FinSim-MAPPO
**Real-Time Financial Network Simulation & Visualization System**

## 1. Purpose

FinSim-MAPPO is a real-time financial network simulator that visualizes strategic decision-making and systemic risk propagation across interconnected financial institutions.

**This system is not:**
- a CRUD app
- a static dashboard
- a data analytics frontend

**This system is:**
- a live simulation
- an event-driven network model
- a visual explanation engine for systemic behavior

## 2. Mental Model (MANDATORY)

> **Backend = financial physics engine**  
> **Frontend = microscope over a living system**

- The backend computes reality.
- The frontend observes and animates it.
- The frontend never infers outcomes.
- The backend never cares about visuals.

## 3. High-Level Architecture

```
Python Simulation Engine (authoritative)
        ↓
FastAPI WebSocket (state + events)
        ↓
React Frontend (rendering only)
        ↓
Cytoscape.js (graph + animations)
```

## 4. Core Principles

- Backend is the single source of truth
- State is immutable
- Events are ephemeral
- State always overrides animation
- Frontend never recalculates anything

**Violating any of these breaks the model.**

## 5. Backend Responsibilities

**The backend must:**
- Advance the simulation timestep
- Maintain the full financial state
- Resolve defaults and contagion
- Detect strategic actions
- Emit:
  - full state snapshots
  - event messages
  - system-level metrics

**The backend must not:**
- Know UI layout
- Know animation timing
- Store frontend state
- Optimize for rendering

## 6. Frontend Responsibilities

**The frontend must:**
- Render the network graph
- Animate events
- Display metrics
- Provide simulation controls
- Explain what is happening visually

**The frontend must not:**
- Recalculate balances
- Infer defaults
- Predict outcomes
- Mutate simulation logic


## 7. Graph Model

### 7.1 Nodes (Banks)

Each node represents one financial institution.

**Node Properties**
```json
{
  "id": "12",
  "tier": 1,
  "capital_ratio": 0.14,
  "stress": 0.62,
  "status": "active",
  "last_updated_timestep": 42
}
```

**Visual Encoding Rules**

| Property | Visual |
|----------|--------|
| tier | node size |
| capital_ratio | color (green → red) |
| stress | glow / intensity |
| status = defaulted | greyed / faded |

### 7.2 Edges (Credit Exposure)

Edges represent directed credit exposure.

**Edge Properties**
```json
{
  "source": "12",
  "target": "8",
  "weight": 40,
  "type": "credit"
}
```

**Visual Encoding Rules**

| Property | Visual |
|----------|--------|
| weight | thickness |
| direction | arrow |
| type | style (solid) |

Edges are static unless animated by events.

## 8. Event Model (CRITICAL CONCEPT)

**Events are visual explanations, not state.**

Events:
- are time-ordered
- are short-lived
- never mutate state
- may occur concurrently

> **State is authoritative.**  
> **Events are illustrative.**

### 8.1 Supported Event Types

**LEND**
- Credit extended from one bank to another
- Visual: moving pulse along edge

**HOARD**
- Liquidity withheld
- Visual: node contraction / lock ring

**FIRE_SALE**
- Forced asset liquidation
- Visual: expanding ripple from node

**PRICE_DROP**
- Market-level shock
- Visual: subtle global pulse or dim

**DEFAULT**
- Structural failure
- Visual: node collapse + edge fade

### 8.2 Event Schema

```json
{
  "event_id": "evt-42-7",
  "event_type": "LEND",
  "from": 12,
  "to": 8,
  "amount": 40,
  "timestamp": 42
}
```

**Rules:**
- Each event triggers one animation
- Animations auto-remove
- Events never block state updates
- State overrides animation if conflict occurs


## 9. WebSocket Contract

**Endpoint:** `/ws/simulation`

WebSocket is **server → client only**.

### 9.1 STATE_UPDATE

```json
{
  "type": "STATE_UPDATE",
  "schema_version": "1.0",
  "payload": {
    "timestep": 42,
    "nodes": [...],
    "edges": [...]
  }
}
```

**Rules:**
- Full snapshot
- Immutable
- Always authoritative

### 9.2 EVENT

```json
{
  "type": "EVENT",
  "payload": {
    "event_id": "evt-42-7",
    "event_type": "FIRE_SALE",
    "from": 9,
    "timestamp": 42
  }
}
```

**Rules:**
- Ephemeral
- Best-effort delivery
- Never authoritative

### 9.3 METRICS_UPDATE

```json
{
  "type": "METRICS_UPDATE",
  "payload": {
    "liquidity": 0.72,
    "default_rate": 0.08,
    "equilibrium_score": 0.81
  }
}
```

## 10. Frontend Control API (REST)

Frontend controls simulation lifecycle via REST.

```
POST /control/play
POST /control/pause
POST /control/step
POST /control/reset
```

**Rules:**
- No WebSocket commands from frontend
- WebSocket is streaming only

## 11. Frontend Rendering Loop

```
receive STATE_UPDATE
→ update node + edge properties
→ do NOT relayout graph

receive EVENT
→ dispatch animation
→ auto-remove animation

repeat
```

**Important:**
- Animations must never block rendering
- State always overrides visuals
- Frontend must tolerate out-of-order events

## 12. Performance Constraints

| Constraint | Limit |
|------------|-------|
| Nodes | ≤ 100 |
| Edges | ≤ 500 |
| Events | ≤ 10 / second |
| Layout recompute | NEVER during run |
| FPS | No drops |

## 13. Frontend Tech Stack (STRICT)

**Allowed:**
- React
- Cytoscape.js
- Native WebSocket
- Minimal state store (Context / Zustand)

**Not allowed:**
- D3 for layout
- Backend polling
- Derived calculations
- UI-side inference

## 14. Backend Structure (Reference)

```
backend/
 ├─ simulator/
 │   ├─ network.py
 │   ├─ pricing.py
 │   ├─ clearing.py
 │   ├─ agents.py
 │   └─ step.py
 │
 ├─ api/
 │   ├─ websocket.py
 │   ├─ control.py
 │   └─ schemas.py
 │
 └─ main.py
```

## 15. Non-Goals

This system intentionally does not include:
- Authentication
- Persistence
- Databases
- Production hardening
- Historical replay

It is a simulation and visualization tool.

## 16. What This System Must Demonstrate

A user watching the graph should clearly see:
- How local decisions affect neighbors
- How risk propagates through the network
- Why liquidity stabilizes systems
- Why hoarding and fire sales cause collapse
- The difference between equilibrium and failure

**If the user can understand what happened without reading numbers, the system is correct.**

## 17. Final Principle

> **Clarity > cleverness**  
> **State > animation**  
> **The graph is the main character**