#!/usr/bin/env python
"""
FinSim-MAPPO: Main Entry Point

Network-Based Multi-Agent Reinforcement Learning System
for Financial Contagion and Stability Analysis
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import load_config, set_seed, get_device


def train(args):
    """Run training."""
    from src.learning import MAPPOTrainer, TrainingConfig
    
    config = load_config(args.config)
    
    training_config = TrainingConfig(
        num_banks=config['network']['num_banks'],
        episode_length=config['simulation']['episode_length'],
        num_episodes=args.episodes or config['simulation']['num_episodes'],
        parallel_envs=config['simulation']['parallel_envs'],
        checkpoint_interval=config['simulation']['checkpoint_interval'],
        actor_hidden_dims=config['mappo']['actor_hidden_dims'],
        critic_hidden_dims=config['mappo']['critic_hidden_dims'],
        learning_rate=config['mappo']['learning_rate'],
        gamma=config['mappo']['gamma'],
        gae_lambda=config['mappo']['gae_lambda'],
        clip_epsilon=config['mappo']['clip_epsilon'],
        entropy_coef_start=config['mappo']['entropy_coef_start'],
        entropy_coef_end=config['mappo']['entropy_coef_end'],
        entropy_decay_steps=config['mappo']['entropy_decay_steps'],
        max_grad_norm=config['mappo']['max_grad_norm'],
        batch_size=config['mappo']['batch_size'],
        update_epochs=config['mappo']['update_epochs'],
        curriculum_enabled=True,
        curriculum_stages=config['curriculum']['stages'],
        agent_freeze_fraction=config['training']['agent_freeze_fraction'],
        freeze_interval=config['training']['freeze_interval'],
        warmup_episodes=config['training']['warmup_episodes'],
        save_dir=config['models']['save_dir'],
        log_dir=config['logging']['log_dir'],
        device=str(get_device()),
        seed=args.seed or config['network']['seed']
    )
    
    print("=" * 60)
    print("FinSim-MAPPO Training")
    print("=" * 60)
    print(f"Number of banks: {training_config.num_banks}")
    print(f"Episodes: {training_config.num_episodes}")
    print(f"Device: {training_config.device}")
    print("=" * 60)
    
    trainer = MAPPOTrainer(training_config)
    trainer.train(progress_bar=True)
    
    # Evaluate
    print("\nEvaluating trained agents...")
    eval_results = trainer.evaluate(num_episodes=20, deterministic=True)
    
    print("\nEvaluation Results:")
    print(f"  Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"  Mean Default Rate: {eval_results['mean_default_rate']:.2%}")
    print(f"  Mean Episode Length: {eval_results['mean_length']:.1f}")


def simulate(args):
    """Run simulation without training."""
    from src.environment import FinancialEnvironment, EnvConfig
    from src.agents.baseline_agents import create_baseline_agent, AgentConfig
    from src.scenarios import ScenarioEngine
    from src.analytics import RiskAnalyzer
    import numpy as np
    
    config = load_config(args.config)
    set_seed(args.seed or config['network']['seed'])
    
    # Create environment
    env_config = EnvConfig(
        num_banks=config['network']['num_banks'],
        episode_length=args.steps,
        seed=args.seed or config['network']['seed']
    )
    env = FinancialEnvironment(env_config)
    
    # Create agents
    agents = {}
    for i in range(env.num_agents):
        agent_config = AgentConfig(
            agent_id=i,
            observation_dim=FinancialEnvironment.OBS_DIM,
            action_dim=FinancialEnvironment.ACTION_DIM
        )
        agents[i] = create_baseline_agent(args.agent_type, agent_config)
    
    # Set up scenario
    scenario_engine = ScenarioEngine()
    scenario_engine.set_scenario(args.scenario)
    
    # Run simulation
    print("=" * 60)
    print("FinSim-MAPPO Simulation")
    print("=" * 60)
    print(f"Scenario: {args.scenario}")
    print(f"Agent Type: {args.agent_type}")
    print(f"Steps: {args.steps}")
    print("=" * 60)
    
    observations, global_state = env.reset()
    total_rewards = {i: 0.0 for i in range(env.num_agents)}
    
    for step in range(args.steps):
        # Generate shocks
        shocks = scenario_engine.generate_shocks(step, args.steps, env.num_agents)
        if shocks['market']:
            env.apply_scenario_shock(**shocks['market'], bank_shocks=shocks['banks'])
        
        # Get actions
        actions = {}
        for agent_id, agent in agents.items():
            actions[agent_id] = agent.select_action(observations[agent_id])
        
        # Step
        result = env.step(actions)
        
        for agent_id, reward in result.rewards.items():
            total_rewards[agent_id] += reward
        
        observations = result.observations
        
        # Print progress
        if (step + 1) % 20 == 0:
            stats = env.network.get_network_stats()
            print(f"Step {step + 1}/{args.steps} | "
                  f"Defaults: {stats.num_defaulted} | "
                  f"Avg CR: {stats.avg_capital_ratio:.2%} | "
                  f"Price: {env.market.current_price:.3f}")
        
        if all(result.dones.values()):
            break
    
    # Final report
    print("\n" + "=" * 60)
    print("Simulation Complete")
    print("=" * 60)
    
    stats = env.network.get_network_stats()
    print(f"Total Steps: {env.current_step}")
    print(f"Final Defaults: {stats.num_defaulted}/{env.num_agents}")
    print(f"Average Reward: {np.mean(list(total_rewards.values())):.2f}")
    print(f"Final Asset Price: {env.market.current_price:.3f}")
    print(f"Final Avg Capital Ratio: {stats.avg_capital_ratio:.2%}")
    
    # Risk analysis
    risk_analyzer = RiskAnalyzer()
    network = env.network
    
    report = risk_analyzer.analyze(
        exposure_matrix=network.liability_matrix.T,
        equity_vector=np.array([b.balance_sheet.equity for b in network.banks.values()]),
        cash_vector=np.array([b.balance_sheet.cash for b in network.banks.values()]),
        liability_vector=np.array([b.balance_sheet.total_liabilities for b in network.banks.values()]),
        capital_ratios=np.array([b.capital_ratio for b in network.banks.values()]),
        graph=network.graph
    )
    
    print(f"\nRisk Metrics:")
    print(f"  DebtRank: {report.debt_rank:.4f}")
    print(f"  Systemic Risk Index: {report.systemic_risk_index:.4f}")
    print(f"  Liquidity Index: {report.liquidity_index:.4f}")
    print(f"  Stress Index: {report.stress_index:.4f}")
    print(f"  Systemically Important Banks: {report.systemically_important_banks}")


def serve(args):
    """Start the API server."""
    import uvicorn
    from api.main import app
    
    config = load_config(args.config)
    
    host = args.host or config['api']['host']
    port = args.port or config['api']['port']
    
    print("=" * 60)
    print("FinSim-MAPPO API Server")
    print("=" * 60)
    print(f"Starting server at http://{host}:{port}")
    print("=" * 60)
    
    uvicorn.run(app, host=host, port=port, reload=args.reload)


def evaluate(args):
    """Evaluate a trained model."""
    from src.learning import MAPPOTrainer, TrainingConfig
    
    config = load_config(args.config)
    
    training_config = TrainingConfig(
        num_banks=config['network']['num_banks'],
        save_dir=config['models']['save_dir'],
        device=str(get_device())
    )
    
    trainer = MAPPOTrainer(training_config)
    trainer.load_checkpoint(args.model)
    
    print("=" * 60)
    print(f"Evaluating Model: {args.model}")
    print("=" * 60)
    
    results = trainer.evaluate(
        num_episodes=args.episodes,
        deterministic=True
    )
    
    print("\nEvaluation Results:")
    print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Mean Default Rate: {results['mean_default_rate']:.2%}")
    print(f"  Mean Episode Length: {results['mean_length']:.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="FinSim-MAPPO: Financial Network Simulation with Multi-Agent RL"
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train MAPPO agents')
    train_parser.add_argument('--episodes', '-e', type=int, help='Number of training episodes')
    train_parser.add_argument('--seed', '-s', type=int, help='Random seed')
    
    # Simulate command
    sim_parser = subparsers.add_parser('simulate', help='Run simulation')
    sim_parser.add_argument('--steps', type=int, default=100, help='Number of simulation steps')
    sim_parser.add_argument('--scenario', type=str, default='normal', 
                           choices=['normal', 'liquidity_crisis', 'asset_crash', 'systemic'],
                           help='Scenario to run')
    sim_parser.add_argument('--agent-type', type=str, default='rule_based',
                           choices=['random', 'rule_based', 'myopic', 'greedy', 'conservative'],
                           help='Type of baseline agent')
    sim_parser.add_argument('--seed', '-s', type=int, help='Random seed')
    
    # Serve command
    serve_parser = subparsers.add_parser('serve', help='Start API server')
    serve_parser.add_argument('--host', type=str, help='Server host')
    serve_parser.add_argument('--port', '-p', type=int, help='Server port')
    serve_parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    eval_parser.add_argument('--model', '-m', type=str, required=True, help='Model checkpoint name')
    eval_parser.add_argument('--episodes', '-e', type=int, default=20, help='Evaluation episodes')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train(args)
    elif args.command == 'simulate':
        simulate(args)
    elif args.command == 'serve':
        serve(args)
    elif args.command == 'evaluate':
        evaluate(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
