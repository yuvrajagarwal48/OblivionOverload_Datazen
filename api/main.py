"""
FastAPI Backend for FinSim-MAPPO
"""

import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import FinancialEnvironment, EnvConfig
from src.agents import MAPPOAgent, MAPPOConfig, BaseAgent
from src.agents.baseline_agents import create_baseline_agent, AgentConfig
from src.scenarios import ScenarioEngine, Scenario, ScenarioType
from src.analytics import RiskAnalyzer
from src.decision_support import DecisionSupport
from src.learning import MAPPOTrainer, TrainingConfig
from src.utils import load_config


# Pydantic Models for API
class SimulationConfig(BaseModel):
    """Configuration for a new simulation."""
    num_banks: int = Field(default=30, ge=5, le=100)
    episode_length: int = Field(default=100, ge=10, le=1000)
    scenario: str = Field(default="normal")
    seed: Optional[int] = None


class ActionRequest(BaseModel):
    """Request for executing agent actions."""
    actions: Dict[int, List[float]]


class WhatIfRequest(BaseModel):
    """Request for what-if analysis."""
    bank_id: int
    action: List[float] = Field(..., min_length=4, max_length=4)


class TrainRequest(BaseModel):
    """Request for starting training."""
    num_episodes: int = Field(default=1000, ge=10)
    checkpoint_interval: int = Field(default=50, ge=10)
    scenario: str = Field(default="normal")


class ScenarioRequest(BaseModel):
    """Request for setting a scenario."""
    scenario_name: str
    custom_params: Optional[Dict[str, float]] = None


class ShockRequest(BaseModel):
    """Request for applying a shock."""
    price_shock: float = Field(default=0.0, ge=-1.0, le=1.0)
    volatility_shock: float = Field(default=0.0, ge=0.0, le=1.0)
    liquidity_shock: float = Field(default=0.0, ge=0.0, le=1.0)
    bank_shocks: Optional[Dict[int, float]] = None


# Global state
class SimulationState:
    """Manages simulation state."""
    
    def __init__(self):
        self.env: Optional[FinancialEnvironment] = None
        self.agents: Dict[int, BaseAgent] = {}
        self.scenario_engine: Optional[ScenarioEngine] = None
        self.risk_analyzer: Optional[RiskAnalyzer] = None
        self.decision_support: Optional[DecisionSupport] = None
        self.trainer: Optional[MAPPOTrainer] = None
        self.current_observations: Dict[int, np.ndarray] = {}
        self.global_state: Optional[np.ndarray] = None
        self.is_training: bool = False
        self.training_progress: float = 0.0
    
    def initialize(self, config: SimulationConfig):
        """Initialize simulation with given config."""
        env_config = EnvConfig(
            num_banks=config.num_banks,
            episode_length=config.episode_length,
            seed=config.seed
        )
        
        self.env = FinancialEnvironment(env_config)
        self.scenario_engine = ScenarioEngine(seed=config.seed)
        self.risk_analyzer = RiskAnalyzer()
        
        # Initialize agents (use baseline agents by default)
        self.agents = {}
        for i in range(config.num_banks):
            agent_config = AgentConfig(
                agent_id=i,
                observation_dim=FinancialEnvironment.OBS_DIM,
                action_dim=FinancialEnvironment.ACTION_DIM,
                seed=config.seed
            )
            self.agents[i] = create_baseline_agent('rule_based', agent_config)
        
        # Set scenario
        self.scenario_engine.set_scenario(config.scenario)
        
        # Reset environment
        self.current_observations, self.global_state = self.env.reset()
        
        # Initialize decision support
        self.decision_support = DecisionSupport(
            self.env, self.agents, horizon=10, num_simulations=10
        )
    
    def is_initialized(self) -> bool:
        return self.env is not None


# Create global state
state = SimulationState()


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title="FinSim-MAPPO API",
        description="Network-Based Multi-Agent RL System for Financial Stability Analysis",
        version="1.0.0"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app


app = create_app()


# Health check
@app.get("/health")
async def health_check():
    """Check API health."""
    return {
        "status": "healthy",
        "initialized": state.is_initialized(),
        "timestamp": datetime.now().isoformat()
    }


# Simulation endpoints
@app.post("/simulation/init")
async def init_simulation(config: SimulationConfig):
    """Initialize a new simulation."""
    try:
        state.initialize(config)
        return {
            "status": "initialized",
            "num_banks": config.num_banks,
            "scenario": config.scenario,
            "message": "Simulation initialized successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simulation/reset")
async def reset_simulation():
    """Reset the current simulation."""
    if not state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    state.current_observations, state.global_state = state.env.reset()
    for agent in state.agents.values():
        agent.reset()
    
    return {
        "status": "reset",
        "step": 0,
        "message": "Simulation reset successfully"
    }


@app.post("/simulation/step")
async def step_simulation(request: Optional[ActionRequest] = None):
    """Execute one simulation step."""
    if not state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    # Get actions
    if request and request.actions:
        actions = {int(k): np.array(v) for k, v in request.actions.items()}
    else:
        # Use agent policies
        actions = {}
        for agent_id, agent in state.agents.items():
            obs = state.current_observations[agent_id]
            actions[agent_id] = agent.select_action(obs, deterministic=True)
    
    # Apply scenario shocks
    shocks = state.scenario_engine.generate_shocks(
        state.env.current_step,
        state.env.config.episode_length,
        state.env.num_agents
    )
    
    if shocks['market']:
        state.env.apply_scenario_shock(**shocks['market'], bank_shocks=shocks['banks'])
    
    # Step environment
    result = state.env.step(actions)
    
    # Update state
    state.current_observations = result.observations
    state.global_state = result.global_state
    
    # Update agent stats
    for agent_id, reward in result.rewards.items():
        state.agents[agent_id].update_stats(reward)
    
    return {
        "step": state.env.current_step,
        "rewards": {int(k): float(v) for k, v in result.rewards.items()},
        "done": any(result.dones.values()),
        "network_stats": result.network_stats.to_dict(),
        "market_state": result.market_state.to_dict()
    }


@app.post("/simulation/run")
async def run_simulation(num_steps: int = 100):
    """Run simulation for multiple steps."""
    if not state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    total_rewards = {i: 0.0 for i in range(state.env.num_agents)}
    steps_completed = 0
    
    for _ in range(num_steps):
        # Get actions from agents
        actions = {}
        for agent_id, agent in state.agents.items():
            obs = state.current_observations[agent_id]
            actions[agent_id] = agent.select_action(obs, deterministic=True)
        
        # Apply scenario shocks
        shocks = state.scenario_engine.generate_shocks(
            state.env.current_step,
            state.env.config.episode_length,
            state.env.num_agents
        )
        
        if shocks['market']:
            state.env.apply_scenario_shock(**shocks['market'], bank_shocks=shocks['banks'])
        
        # Step
        result = state.env.step(actions)
        
        state.current_observations = result.observations
        state.global_state = result.global_state
        
        for agent_id, reward in result.rewards.items():
            total_rewards[agent_id] += reward
        
        steps_completed += 1
        
        if any(result.dones.values()):
            break
    
    return {
        "steps_completed": steps_completed,
        "total_rewards": {int(k): float(v) for k, v in total_rewards.items()},
        "final_step": state.env.current_step,
        "network_stats": state.env.network.get_network_stats().to_dict()
    }


# Metrics endpoints
@app.get("/metrics")
async def get_metrics():
    """Get current system metrics."""
    if not state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    # Network stats
    network_stats = state.env.network.get_network_stats()
    
    # Market state
    market_state = state.env.market.get_state()
    
    # Bank summaries
    bank_summaries = {}
    for bank_id, bank in state.env.network.banks.items():
        bank_summaries[bank_id] = {
            "equity": bank.balance_sheet.equity,
            "capital_ratio": bank.capital_ratio,
            "cash": bank.balance_sheet.cash,
            "status": bank.status.value,
            "tier": bank.tier
        }
    
    return {
        "step": state.env.current_step,
        "network": network_stats.to_dict(),
        "market": market_state.to_dict(),
        "banks": bank_summaries,
        "scenario": state.scenario_engine.get_scenario_summary()
    }


@app.get("/metrics/risk")
async def get_risk_metrics():
    """Get comprehensive risk analysis."""
    if not state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    # Build inputs for risk analysis
    network = state.env.network
    
    equity_vector = np.array([b.balance_sheet.equity for b in network.banks.values()])
    cash_vector = np.array([b.balance_sheet.cash for b in network.banks.values()])
    liability_vector = np.array([b.balance_sheet.total_liabilities for b in network.banks.values()])
    capital_ratios = np.array([b.capital_ratio for b in network.banks.values()])
    
    # Run analysis
    report = state.risk_analyzer.analyze(
        exposure_matrix=network.liability_matrix.T,  # Transpose for exposure
        equity_vector=equity_vector,
        cash_vector=cash_vector,
        liability_vector=liability_vector,
        capital_ratios=capital_ratios,
        graph=network.graph
    )
    
    return report.to_dict()


@app.get("/metrics/bank/{bank_id}")
async def get_bank_metrics(bank_id: int):
    """Get detailed metrics for a specific bank."""
    if not state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    if bank_id not in state.env.network.banks:
        raise HTTPException(status_code=404, detail=f"Bank {bank_id} not found")
    
    bank = state.env.network.banks[bank_id]
    
    # Get centrality metrics
    centrality = state.env.network.get_centrality_metrics()
    
    return {
        "bank_id": bank_id,
        "tier": bank.tier,
        "status": bank.status.value,
        "balance_sheet": bank.balance_sheet.to_dict(),
        "capital_ratio": bank.capital_ratio,
        "is_solvent": bank.is_solvent,
        "is_liquid": bank.is_liquid,
        "excess_cash": bank.excess_cash,
        "centrality": centrality.get(bank_id, {}),
        "neighbors": state.env.network.get_neighbors(bank_id),
        "creditors": state.env.network.get_creditors(bank_id),
        "debtors": state.env.network.get_debtors(bank_id)
    }


# Scenario endpoints
@app.get("/scenarios")
async def list_scenarios():
    """List available scenarios."""
    if not state.is_initialized():
        # Return default scenarios
        from src.scenarios.scenario_engine import PREDEFINED_SCENARIOS
        return {
            "scenarios": [
                {"name": name, "type": s.scenario_type.value, "description": s.description}
                for name, s in PREDEFINED_SCENARIOS.items()
            ]
        }
    
    return {
        "scenarios": state.scenario_engine.list_scenarios(),
        "current": state.scenario_engine.get_scenario_summary()
    }


@app.post("/scenarios/set")
async def set_scenario(request: ScenarioRequest):
    """Set the current scenario."""
    if not state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    success = state.scenario_engine.set_scenario(request.scenario_name)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Scenario '{request.scenario_name}' not found")
    
    return {
        "status": "success",
        "scenario": state.scenario_engine.get_scenario_summary()
    }


@app.post("/scenarios/shock")
async def apply_shock(request: ShockRequest):
    """Apply an ad-hoc shock to the system."""
    if not state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    bank_shocks = {int(k): v for k, v in request.bank_shocks.items()} if request.bank_shocks else None
    
    state.env.apply_scenario_shock(
        price_shock=request.price_shock,
        volatility_shock=request.volatility_shock,
        liquidity_shock=request.liquidity_shock,
        bank_shocks=bank_shocks
    )
    
    return {
        "status": "shock_applied",
        "market_state": state.env.market.get_state().to_dict()
    }


# Decision support endpoints
@app.post("/what_if")
async def what_if_analysis(request: WhatIfRequest):
    """Perform what-if analysis for a proposed action."""
    if not state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    if request.bank_id not in state.agents:
        raise HTTPException(status_code=404, detail=f"Bank {request.bank_id} not found")
    
    action = np.array(request.action)
    
    result = state.decision_support.evaluate_what_if(
        bank_id=request.bank_id,
        action=action,
        observations=state.current_observations
    )
    
    return result


@app.get("/recommendations/{bank_id}")
async def get_recommendation(bank_id: int):
    """Get action recommendation for a specific bank."""
    if not state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    if bank_id not in state.agents:
        raise HTTPException(status_code=404, detail=f"Bank {bank_id} not found")
    
    recommendation = state.decision_support.get_recommendation(
        bank_id=bank_id,
        observations=state.current_observations
    )
    
    return recommendation.to_dict()


@app.get("/recommendations")
async def get_all_recommendations():
    """Get action recommendations for all banks."""
    if not state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    report = state.decision_support.generate_report(state.current_observations)
    report['timestamp'] = datetime.now().isoformat()
    
    return report


# Training endpoints
@app.post("/train/start")
async def start_training(request: TrainRequest, background_tasks: BackgroundTasks):
    """Start training in the background."""
    if state.is_training:
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    def run_training():
        state.is_training = True
        state.training_progress = 0.0
        
        try:
            config = TrainingConfig(
                num_banks=state.env.num_agents if state.env else 30,
                num_episodes=request.num_episodes,
                checkpoint_interval=request.checkpoint_interval
            )
            
            state.trainer = MAPPOTrainer(config)
            state.trainer.train(progress_bar=False)
            
            # Update agents with trained policies
            for agent_id, trained_agent in state.trainer.agents.items():
                state.agents[agent_id] = trained_agent
            
        finally:
            state.is_training = False
            state.training_progress = 1.0
    
    background_tasks.add_task(run_training)
    
    return {
        "status": "training_started",
        "num_episodes": request.num_episodes
    }


@app.get("/train/status")
async def get_training_status():
    """Get current training status."""
    return {
        "is_training": state.is_training,
        "progress": state.training_progress,
        "current_episode": state.trainer.current_episode if state.trainer else 0
    }


@app.post("/train/stop")
async def stop_training():
    """Stop current training."""
    if not state.is_training:
        raise HTTPException(status_code=400, detail="No training in progress")
    
    # Note: This requires cooperative stopping in the training loop
    state.is_training = False
    
    return {"status": "stop_requested"}


# Model endpoints
@app.post("/models/save")
async def save_model(name: str = "manual_save"):
    """Save current model state."""
    if not state.trainer:
        raise HTTPException(status_code=400, detail="No trained model available")
    
    state.trainer.save_checkpoint(name)
    
    return {"status": "saved", "name": name}


@app.post("/models/load")
async def load_model(name: str):
    """Load a saved model."""
    if not state.trainer:
        # Create trainer first
        config = TrainingConfig(
            num_banks=state.env.num_agents if state.env else 30
        )
        state.trainer = MAPPOTrainer(config)
    
    try:
        state.trainer.load_checkpoint(name)
        
        # Update agents
        for agent_id, trained_agent in state.trainer.agents.items():
            state.agents[agent_id] = trained_agent
        
        return {"status": "loaded", "name": name}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Failed to load model: {str(e)}")


@app.get("/models")
async def list_models():
    """List available saved models."""
    models_dir = "models"
    if not os.path.exists(models_dir):
        return {"models": []}
    
    models = []
    for name in os.listdir(models_dir):
        path = os.path.join(models_dir, name)
        if os.path.isdir(path):
            config_path = os.path.join(path, "config.json")
            if os.path.exists(config_path):
                models.append({
                    "name": name,
                    "path": path
                })
    
    return {"models": models}


# Network topology endpoint
@app.get("/network/topology")
async def get_network_topology():
    """Get network topology for visualization."""
    if not state.is_initialized():
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    network = state.env.network
    
    nodes = []
    for bank_id, bank in network.banks.items():
        nodes.append({
            "id": bank_id,
            "tier": bank.tier,
            "equity": bank.balance_sheet.equity,
            "status": bank.status.value,
            "capital_ratio": bank.capital_ratio
        })
    
    edges = []
    for u, v in network.graph.edges():
        weight = network.get_exposure(u, v)
        edges.append({
            "source": u,
            "target": v,
            "weight": weight
        })
    
    return {
        "nodes": nodes,
        "edges": edges,
        "stats": network.get_network_stats().to_dict()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
