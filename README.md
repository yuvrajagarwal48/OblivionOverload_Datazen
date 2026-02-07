# FinSim-MAPPO

## Network-Based Multi-Agent Reinforcement Learning System for Financial Contagion and Stability Analysis

A comprehensive simulation framework for modeling strategic interactions among financial institutions using game-theoretic multi-agent reinforcement learning.

## Overview

FinSim-MAPPO integrates:

- **Financial Environment**: Physics-based simulator with balance-sheet accounting, debt clearing (Eisenberg-Noe), and asset pricing
- **Learning Framework**: Multi-Agent Proximal Policy Optimization (MAPPO) with Centralized Training, Decentralized Execution (CTDE)
- **Incentive Design**: Reward shaping for aligning individual profit-seeking with system stability
- **Analytics**: Comprehensive risk metrics including DebtRank, systemic risk indices, and contagion analysis
- **Decision Support**: Lookahead simulation for action recommendations

## Project Structure

```
finsim_mappo/
├── config/
│   └── config.yaml          # Configuration file
├── src/
│   ├── environment/         # Financial simulation environment
│   │   ├── network.py       # Network topology (Barabási-Albert)
│   │   ├── bank.py          # Bank entity with balance sheet
│   │   ├── clearing.py      # Eisenberg-Noe clearing mechanism
│   │   ├── market.py        # Asset pricing and fire sales
│   │   ├── exchange.py      # Exchange nodes (trade routing)
│   │   ├── ccp.py           # Central Counterparty (margins, netting)
│   │   ├── infrastructure.py # Infrastructure router
│   │   └── financial_env.py # Main environment
│   ├── agents/              # Agent implementations
│   │   ├── base_agent.py    # Abstract base agent
│   │   ├── baseline_agents.py # Rule-based, Myopic, Greedy, etc.
│   │   └── mappo_agent.py   # MAPPO agent with actor network
│   ├── learning/            # Training infrastructure
│   │   ├── ppo.py           # PPO algorithm
│   │   ├── trainer.py       # MAPPO trainer with curriculum
│   │   └── state_compression.py # Global state compression for critic
│   ├── scenarios/           # Scenario definitions
│   │   └── scenario_engine.py
│   ├── analytics/           # Risk analysis
│   │   └── risk_metrics.py  # DebtRank, systemic risk
│   ├── decision_support/    # Recommendations
│   │   └── lookahead.py     # What-if analysis
│   ├── evaluation/          # Formal evaluation pipeline
│   │   └── pipeline.py      # Baseline comparison, multi-seed
│   ├── models/              # Model lifecycle management
│   │   └── lifecycle.py     # Checkpointing, versioning
│   ├── logging/             # Comprehensive logging
│   │   └── __init__.py      # Training/environment loggers
│   ├── visualization/       # Visualization tools
│   │   └── plots.py         # Simulation, network, infrastructure viz
│   └── utils/               # Utilities
│       ├── helpers.py
│       └── stability.py     # Numerical stability safeguards
├── api/                     # FastAPI backend
│   └── main.py
├── models/                  # Saved models
├── logs/                    # Training logs
├── run.py                   # Main entry point
├── demo.py                  # Basic demo with visualization
├── demo_infrastructure.py   # Demo with infrastructure layer
└── requirements.txt
```

## Installation

```bash
# Clone the repository
cd Datazen

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Run a Simulation

```bash
# Basic simulation with default settings
python run.py simulate

# Simulation with specific scenario
python run.py simulate --scenario liquidity_crisis --steps 200

# Use different agent types
python run.py simulate --agent-type myopic --scenario asset_crash
```

### 2. Train MAPPO Agents

```bash
# Train with default configuration
python run.py train

# Train with custom episodes
python run.py train --episodes 2000 --seed 42
```

### 3. Start API Server

```bash
# Start the FastAPI server
python run.py serve

# With custom host/port
python run.py serve --host 0.0.0.0 --port 8080 --reload
```

### 4. Evaluate Trained Model

```bash
python run.py evaluate --model final --episodes 50
```

### 5. Run Infrastructure Demo

```bash
# Run with infrastructure layer (exchanges, CCPs)
python demo_infrastructure.py --scenario liquidity_crisis

# Customize infrastructure
python demo_infrastructure.py --banks 30 --exchanges 2 --ccps 1

# Run scenario comparison
python demo.py --compare
```

### 6. Run Formal Evaluation Pipeline

```bash
# Quick baseline comparison
python -c "from src.evaluation import run_quick_evaluation; run_quick_evaluation()"
```

## Infrastructure Layer

FinSim-MAPPO now includes a full financial infrastructure layer:

### Exchanges

Exchanges act as trade routing intermediaries:

- **Order Management**: Submit, process, and match orders
- **Congestion Tracking**: Dynamic throughput limits
- **Fee Calculation**: Base fees with congestion multipliers
- **Settlement Delays**: Realistic processing time

```python
from src.environment import Exchange, ExchangeConfig

exchange = Exchange(
    exchange_id=0,
    config=ExchangeConfig(
        max_throughput=1000,
        base_fee_rate=0.001,
        congestion_threshold=0.7
    )
)
```

### Central Counterparties (CCPs)

CCPs provide clearing and risk management:

- **Margin System**: Initial and variation margins
- **Multilateral Netting**: Reduce gross exposures (60% efficiency)
- **Default Waterfall**: 4-layer loss absorption
  1. Defaulter's margin
  2. Default fund contribution
  3. CCP capital (skin-in-the-game)
  4. Mutualization across members

```python
from src.environment import CentralCounterparty

ccp = CentralCounterparty(
    ccp_id=0,
    initial_capital=10_000_000,
    initial_margin_rate=0.1,
    netting_efficiency=0.6
)

# Register members
for bank_id in range(20):
    ccp.register_member(bank_id, initial_collateral=100_000)
```

### Infrastructure Router

Routes all transactions through proper channels:

- **Enforced Path**: Banks → Exchanges → CCPs → Banks
- **No Direct Settlement**: All flows mediated by infrastructure
- **8-Dimensional Observation**: Infrastructure state for agents

```python
from src.environment import InfrastructureRouter

router = InfrastructureRouter(
    exchanges=exchanges,
    ccps=ccps,
    banks=banks
)

# Route transaction
result = router.submit_transaction(
    sender_id=0,
    receiver_id=5,
    amount=100000,
    transaction_type='lending'
)
```

## API Endpoints

| Endpoint                | Method | Description           |
| ----------------------- | ------ | --------------------- |
| `/health`               | GET    | Health check          |
| `/simulation/init`      | POST   | Initialize simulation |
| `/simulation/step`      | POST   | Execute one step      |
| `/simulation/run`       | POST   | Run multiple steps    |
| `/metrics`              | GET    | Get current metrics   |
| `/metrics/risk`         | GET    | Risk analysis         |
| `/metrics/bank/{id}`    | GET    | Bank details          |
| `/scenarios`            | GET    | List scenarios        |
| `/scenarios/set`        | POST   | Set scenario          |
| `/scenarios/shock`      | POST   | Apply shock           |
| `/what_if`              | POST   | What-if analysis      |
| `/recommendations/{id}` | GET    | Get recommendation    |
| `/train/start`          | POST   | Start training        |
| `/network/topology`     | GET    | Network structure     |

## Configuration

Edit `config/config.yaml` to customize:

```yaml
# Network Parameters
network:
  num_banks: 30
  edges_per_node: 2
  core_fraction: 0.2

# MAPPO Hyperparameters
mappo:
  learning_rate: 0.0003
  gamma: 0.99
  clip_epsilon: 0.2

# Scenarios
scenarios:
  normal:
    shock_probability: 0.05
    shock_magnitude: 0.05
  liquidity_crisis:
    shock_probability: 0.3
    liquidity_stress: 0.5
```

## Key Concepts

### Financial Environment

- **Banks**: Entities with balance sheets (cash, assets, liabilities)
- **Network**: Scale-free topology via Barabási-Albert model
- **Clearing**: Eisenberg-Noe algorithm for resolving defaults
- **Market**: Price impact model with fire sale dynamics

### Agent Actions

Each agent outputs a 4-dimensional action:

1. **Lend Ratio**: Fraction of excess cash to lend
2. **Hoard Ratio**: Cash retention
3. **Sell Ratio**: Assets to liquidate
4. **Borrow Request**: Liquidity demand

### Reward Function

```
R = R_profit + μ·R_liquidity - α·I_default - λ·R_system_risk
```

### Training Features

- **Curriculum Learning**: Progressive difficulty
- **Agent Freezing**: Stability through partial updates
- **Entropy Scheduling**: Exploration decay

## Scenarios

| Scenario         | Description                         |
| ---------------- | ----------------------------------- |
| Normal           | Stable market, minimal shocks       |
| Liquidity Crisis | Funding freeze, high stress         |
| Asset Crash      | Sharp price decline, fire sales     |
| Systemic         | Combined crisis, cascading failures |

## Risk Metrics

- **DebtRank**: Systemic importance measure
- **Liquidity Index**: System-wide liquidity
- **Stress Index**: Capital adequacy stress
- **Contagion Depth**: Cascade propagation depth

## Advanced Features

### Global State Compression

For scalable centralized critics with many agents:

```python
from src.learning import GlobalStateCompressor, GlobalStateConfig

compressor = GlobalStateCompressor(
    num_agents=50,
    obs_dim=16,
    config=GlobalStateConfig(
        compressed_dim=64,
        use_attention=True
    )
)
```

### Model Lifecycle Management

Versioned checkpointing with metadata:

```python
from src.models import ModelCheckpointer, ModelRegistry

checkpointer = ModelCheckpointer(checkpoint_dir="outputs/checkpoints")

# Save checkpoint
info = checkpointer.save_checkpoint(
    model_id="mappo_v1",
    agents=agents,
    optimizer_states=optimizers,
    training_state={'episode': 100},
    env_config=env_config,
    agent_config=agent_config,
    metrics={'mean_reward': 5.2}
)

# Load best checkpoint
state_dict, metadata = checkpointer.load_checkpoint("mappo_v1", version="best")
```

### Numerical Stability

Safeguards for training stability:

```python
from src.utils import NumericalStabilityGuard, StabilityConfig

guard = NumericalStabilityGuard(StabilityConfig(
    max_grad_norm=0.5,
    obs_clip=10.0,
    reward_clip=100.0
))

# Safe operations
safe_obs = guard.clip_observations(obs)
grad_norm = guard.clip_gradients(model)
```

### Comprehensive Logging

Training and environment monitoring:

```python
from src.logging import TrainingLogger, EnvironmentLogger, SystemMonitor

# Training logger
train_log = TrainingLogger(log_dir="logs/training")
train_log.log_episode(episode=1, rewards=rewards, losses=losses)

# Environment logger
env_log = EnvironmentLogger(log_dir="logs/environment")
env_log.log_step(step=1, observations=obs, actions=actions)

# System monitor with alerts
monitor = SystemMonitor(alert_thresholds={'default_rate': 0.2})
monitor.check_health(metrics)
```

## Example Usage

### Python API

```python
from src.environment import FinancialEnvironment, EnvConfig
from src.agents.baseline_agents import create_baseline_agent, AgentConfig

# Create environment
env = FinancialEnvironment(EnvConfig(num_banks=20))
observations, global_state = env.reset()

# Create agents
agents = {i: create_baseline_agent('rule_based', AgentConfig(i))
          for i in range(20)}

# Run simulation
for step in range(100):
    actions = {i: agents[i].select_action(observations[i])
               for i in agents}
    result = env.step(actions)
    observations = result.observations
```

### REST API

```bash
# Initialize simulation
curl -X POST http://localhost:8000/simulation/init \
  -H "Content-Type: application/json" \
  -d '{"num_banks": 30, "scenario": "normal"}'

# Run steps
curl -X POST http://localhost:8000/simulation/run?num_steps=50

# Get risk metrics
curl http://localhost:8000/metrics/risk

# Get recommendation
curl http://localhost:8000/recommendations/0
```

## License

MIT License

## Citation

If you use this work, please cite:

```
@software{finsim_mappo,
  title={FinSim-MAPPO: Network-Based Multi-Agent RL for Financial Stability},
  year={2024}
}
```
