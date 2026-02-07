"""
Enhanced Demo Script for FinSim-MAPPO with Visualization
Demonstrates simulation, analysis, and visual output capabilities.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import argparse

from src.environment import FinancialEnvironment, EnvConfig
from src.agents.baseline_agents import create_baseline_agent, AgentConfig
from src.scenarios import ScenarioEngine
from src.analytics import RiskAnalyzer
from src.visualization import (
    SimulationVisualizer, 
    NetworkVisualizer,
    RiskDashboard
)
from src.visualization.plots import (
    print_simulation_header,
    print_step_progress,
    print_simulation_summary
)


def run_demo(scenario: str = "normal", num_banks: int = 20, 
             num_steps: int = 50, visualize: bool = True, 
             save_plots: bool = True, show_plots: bool = True):
    """
    Run a demonstration of the FinSim-MAPPO system with visualization.
    
    Args:
        scenario: Scenario to run ('normal', 'liquidity_crisis', 'asset_crash', 'systemic')
        num_banks: Number of banks in the network
        num_steps: Number of simulation steps
        visualize: Whether to create visualizations
        save_plots: Whether to save plots to files
        show_plots: Whether to display plots interactively
    """
    
    # Print header
    print_simulation_header(scenario, num_banks, num_steps)
    
    # Initialize visualizer
    if visualize:
        sim_viz = SimulationVisualizer(save_dir="outputs/figures")
        net_viz = NetworkVisualizer(save_dir="outputs/figures")
        risk_dashboard = RiskDashboard(save_dir="outputs/figures")
    
    print("  [1/6] 🏗️  Initializing environment...")
    
    # Create environment
    env_config = EnvConfig(
        num_banks=num_banks,
        episode_length=num_steps,
        seed=42
    )
    env = FinancialEnvironment(env_config)
    
    # Reset environment (initializes network and market)
    observations, global_state = env.reset()
    
    print(f"        ✓ Network: {env.network.graph.number_of_edges()} edges, "
          f"{sum(1 for b in env.network.banks.values() if b.tier == 1)} core banks")
    
    # Visualize initial network
    if visualize:
        print("\n  [2/6] 🔗 Visualizing initial network...")
        net_viz.plot_network(
            env.network.graph, 
            env.network.banks,
            title=f"Initial Financial Network ({scenario.title()} Scenario)",
            save=save_plots,
            show=False  # Don't show yet, we'll show all at the end
        )
        
        net_viz.plot_exposure_heatmap(
            env.network.liability_matrix,
            title="Initial Interbank Exposure Matrix",
            save=save_plots,
            show=False
        )
    
    # Create agents
    print("\n  [3/6] 🤖 Creating agents...")
    agents = {}
    agent_types = ['rule_based', 'myopic', 'conservative', 'greedy']
    agent_counts = {t: 0 for t in agent_types}
    
    for i in range(num_banks):
        agent_type = agent_types[i % len(agent_types)]
        agent_config = AgentConfig(
            agent_id=i,
            observation_dim=FinancialEnvironment.OBS_DIM,
            action_dim=FinancialEnvironment.ACTION_DIM,
            seed=42 + i
        )
        agents[i] = create_baseline_agent(agent_type, agent_config)
        agent_counts[agent_type] += 1
    
    print(f"        ✓ Created {len(agents)} agents:")
    for t, c in agent_counts.items():
        print(f"          • {t}: {c} agents")
    
    # Set up scenario
    print(f"\n  [4/6] 📋 Setting scenario: {scenario}")
    scenario_engine = ScenarioEngine(seed=42)
    scenario_engine.set_scenario(scenario)
    
    current_scenario = scenario_engine.current_scenario
    print(f"        ✓ Shock probability: {current_scenario.params.shock_probability:.0%}")
    print(f"        ✓ Shock magnitude: {current_scenario.params.shock_magnitude:.0%}")
    
    # Run simulation
    print(f"\n  [5/6] 🚀 Running simulation...")
    print("  " + "─" * 66)
    
    total_rewards = {i: 0.0 for i in range(num_banks)}
    
    for step in range(num_steps):
        # Apply scenario shocks
        shocks = scenario_engine.generate_shocks(step, num_steps, num_banks)
        if any(v != 0 for v in shocks['market'].values()):
            env.apply_scenario_shock(**shocks['market'])
        
        # Get actions from all agents
        actions = {}
        for agent_id, agent in agents.items():
            actions[agent_id] = agent.select_action(observations[agent_id])
        
        # Step environment
        result = env.step(actions)
        
        # Accumulate rewards
        for agent_id, reward in result.rewards.items():
            total_rewards[agent_id] += reward
        
        observations = result.observations
        
        # Record for visualization
        if visualize:
            bank_equities = {i: b.balance_sheet.equity 
                           for i, b in env.network.banks.items()}
            sim_viz.record_step(
                step=step + 1,
                market_state=result.market_state,
                network_stats=result.network_stats,
                rewards=result.rewards,
                bank_equities=bank_equities
            )
        
        # Print progress
        print_step_progress(
            step + 1, num_steps,
            result.network_stats.num_defaulted,
            result.network_stats.num_stressed,
            result.market_state.asset_price,
            result.network_stats.avg_capital_ratio
        )
        
        if all(result.dones.values()):
            print(f"\n        ⚠️  Episode ended early at step {step + 1}")
            break
    
    print("\n  " + "─" * 66)
    
    # Risk analysis
    print("\n  [6/6] 📊 Analyzing results...")
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
    
    # Get final states
    final_stats = env.network.get_network_stats()
    market_state = env.market.get_state()
    
    # Print summary
    print_simulation_summary(final_stats, market_state, total_rewards, report)
    
    # Create visualizations
    if visualize:
        print("  📈 Generating visualizations...")
        
        # Simulation summary plot
        sim_viz.plot_simulation_summary(
            title=f"Simulation Summary - {scenario.title()} Scenario",
            save=save_plots,
            show=False
        )
        
        # Bank trajectories
        sim_viz.plot_bank_trajectories(
            num_banks=min(10, num_banks),
            save=save_plots,
            show=False
        )
        
        # Final network state
        net_viz.plot_network(
            env.network.graph,
            env.network.banks,
            title=f"Final Network State - {scenario.title()} Scenario",
            save=save_plots,
            show=False
        )
        
        # Risk dashboard
        risk_dashboard.create_dashboard(
            report, final_stats, market_state,
            title=f"Risk Dashboard - {scenario.title()} Scenario",
            save=save_plots,
            show=False
        )
        
        print("\n  ✅ All visualizations saved to outputs/figures/")
        
        if show_plots:
            print("  🖼️  Displaying plots...")
            import matplotlib.pyplot as plt
            plt.show()
    
    return {
        'final_stats': final_stats,
        'market_state': market_state,
        'total_rewards': total_rewards,
        'risk_report': report
    }


def compare_scenarios(visualize: bool = True):
    """Compare different stress scenarios with visualization."""
    
    print("\n" + "═" * 70)
    print("║" + " " * 18 + "SCENARIO COMPARISON ANALYSIS" + " " * 20 + "║")
    print("═" * 70 + "\n")
    
    scenarios = ['normal', 'liquidity_crisis', 'asset_crash', 'systemic']
    results = {}
    
    for scenario_name in scenarios:
        print(f"  Running scenario: {scenario_name}...")
        
        # Create fresh environment
        env_config = EnvConfig(num_banks=20, episode_length=50, seed=42)
        env = FinancialEnvironment(env_config)
        observations, _ = env.reset()
        
        # Create agents
        agents = {i: create_baseline_agent('rule_based', AgentConfig(i)) 
                  for i in range(20)}
        
        # Set scenario
        scenario_engine = ScenarioEngine(seed=42)
        scenario_engine.set_scenario(scenario_name)
        
        # Run
        total_reward = 0.0
        
        for step in range(50):
            shocks = scenario_engine.generate_shocks(step, 50, 20)
            if shocks['market']:
                env.apply_scenario_shock(**shocks['market'])
            
            actions = {i: agents[i].select_action(observations[i]) for i in agents}
            result = env.step(actions)
            total_reward += sum(result.rewards.values())
            observations = result.observations
            
            if all(result.dones.values()):
                break
        
        stats = env.network.get_network_stats()
        results[scenario_name] = {
            'defaults': stats.num_defaulted,
            'stressed': stats.num_stressed,
            'avg_reward': total_reward / 20,
            'final_price': env.market.current_price,
            'avg_cr': stats.avg_capital_ratio,
            'total_exposure': stats.total_exposure
        }
        print(f"    ✓ Completed: {stats.num_defaulted} defaults, "
              f"reward: {total_reward/20:.1f}")
    
    # Print comparison table
    print("\n" + "─" * 90)
    print(f"  {'Scenario':<18} │ {'Defaults':>8} │ {'Stressed':>8} │ "
          f"{'Avg Reward':>10} │ {'Price':>8} │ {'Cap Ratio':>10}")
    print("─" * 90)
    
    for scenario, r in results.items():
        print(f"  {scenario:<18} │ {r['defaults']:>8} │ {r['stressed']:>8} │ "
              f"{r['avg_reward']:>10.1f} │ {r['final_price']:>8.3f} │ "
              f"{r['avg_cr']*100:>9.1f}%")
    
    print("─" * 90)
    
    # Visualization
    if visualize:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        scenarios_list = list(results.keys())
        x = np.arange(len(scenarios_list))
        
        # 1. Defaults & Stressed
        ax1 = axes[0, 0]
        width = 0.35
        ax1.bar(x - width/2, [results[s]['defaults'] for s in scenarios_list], 
                width, label='Defaults', color='#e74c3c')
        ax1.bar(x + width/2, [results[s]['stressed'] for s in scenarios_list], 
                width, label='Stressed', color='#f39c12')
        ax1.set_ylabel('Number of Banks')
        ax1.set_title('Bank Failures by Scenario')
        ax1.set_xticks(x)
        ax1.set_xticklabels([s.replace('_', '\n') for s in scenarios_list])
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. Average Reward
        ax2 = axes[0, 1]
        colors = ['#2ecc71' if r['avg_reward'] > 0 else '#e74c3c' 
                  for r in results.values()]
        ax2.bar(x, [results[s]['avg_reward'] for s in scenarios_list], color=colors)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_ylabel('Average Reward per Agent')
        ax2.set_title('Agent Performance by Scenario')
        ax2.set_xticks(x)
        ax2.set_xticklabels([s.replace('_', '\n') for s in scenarios_list])
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Asset Price
        ax3 = axes[1, 0]
        prices = [results[s]['final_price'] for s in scenarios_list]
        ax3.bar(x, prices, color='#3498db')
        ax3.axhline(y=1.0, color='red', linestyle='--', label='Initial Price')
        ax3.set_ylabel('Final Asset Price')
        ax3.set_title('Asset Price Impact by Scenario')
        ax3.set_xticks(x)
        ax3.set_xticklabels([s.replace('_', '\n') for s in scenarios_list])
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Capital Ratio
        ax4 = axes[1, 1]
        crs = [results[s]['avg_cr'] * 100 for s in scenarios_list]
        colors = ['#2ecc71' if cr >= 8 else '#e74c3c' for cr in crs]
        ax4.bar(x, crs, color=colors)
        ax4.axhline(y=8, color='red', linestyle='--', label='Min Requirement (8%)')
        ax4.set_ylabel('Capital Ratio (%)')
        ax4.set_title('Capital Adequacy by Scenario')
        ax4.set_xticks(x)
        ax4.set_xticklabels([s.replace('_', '\n') for s in scenarios_list])
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        fig.suptitle('Scenario Comparison Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        from pathlib import Path
        save_dir = Path("outputs/figures")
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / "scenario_comparison.png", dpi=150, bbox_inches='tight')
        print(f"\n  📊 Saved comparison chart to outputs/figures/scenario_comparison.png")
        
        plt.show()
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="FinSim-MAPPO Demo with Visualization"
    )
    parser.add_argument(
        '--scenario', '-s',
        type=str,
        default='normal',
        choices=['normal', 'liquidity_crisis', 'asset_crash', 'systemic'],
        help='Scenario to simulate'
    )
    parser.add_argument(
        '--banks', '-b',
        type=int,
        default=20,
        help='Number of banks'
    )
    parser.add_argument(
        '--steps', '-t',
        type=int,
        default=50,
        help='Number of simulation steps'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Run scenario comparison instead of single simulation'
    )
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Disable visualization'
    )
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Save plots but do not display them'
    )
    
    args = parser.parse_args()
    
    if args.compare:
        compare_scenarios(visualize=not args.no_viz)
    else:
        run_demo(
            scenario=args.scenario,
            num_banks=args.banks,
            num_steps=args.steps,
            visualize=not args.no_viz,
            save_plots=True,
            show_plots=not args.no_show
        )


if __name__ == "__main__":
    main()
