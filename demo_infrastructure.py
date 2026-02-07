"""
Enhanced Demo Script with Infrastructure Layer
Demonstrates exchanges, CCPs, and infrastructure routing.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import argparse

from src.environment import (
    FinancialEnvironment, EnvConfig, 
    Exchange, ExchangeNetwork, ExchangeConfig,
    CentralCounterparty, CCPNetwork,
    InfrastructureRouter
)
from src.agents.baseline_agents import create_baseline_agent, AgentConfig
from src.scenarios import ScenarioEngine
from src.analytics import RiskAnalyzer
from src.visualization import (
    SimulationVisualizer, 
    NetworkVisualizer,
    RiskDashboard,
    InfrastructureVisualizer
)
from src.visualization.plots import (
    print_simulation_header,
    print_step_progress,
    print_simulation_summary
)


def run_infrastructure_demo(
    scenario: str = "normal",
    num_banks: int = 20,
    num_steps: int = 50,
    num_exchanges: int = 2,
    num_ccps: int = 1,
    visualize: bool = True,
    save_plots: bool = True,
    show_plots: bool = True
):
    """
    Run demonstration with full infrastructure layer.
    
    Args:
        scenario: Stress scenario
        num_banks: Number of banks
        num_steps: Simulation length
        num_exchanges: Number of exchanges
        num_ccps: Number of CCPs
        visualize: Enable visualization
        save_plots: Save plots to disk
        show_plots: Display plots
    """
    print("\n" + "═" * 70)
    print("║" + " " * 12 + "FINSIM-MAPPO INFRASTRUCTURE DEMO" + " " * 19 + "║")
    print("═" * 70)
    print(f"║  Scenario:     {scenario:<52s}║")
    print(f"║  Banks:        {num_banks:<52d}║")
    print(f"║  Exchanges:    {num_exchanges:<52d}║")
    print(f"║  CCPs:         {num_ccps:<52d}║")
    print(f"║  Steps:        {num_steps:<52d}║")
    print("═" * 70 + "\n")
    
    # Initialize visualizers
    if visualize:
        sim_viz = SimulationVisualizer(save_dir="outputs/figures")
        net_viz = NetworkVisualizer(save_dir="outputs/figures")
        infra_viz = InfrastructureVisualizer(save_dir="outputs/figures")
        risk_dashboard = RiskDashboard(save_dir="outputs/figures")
    
    # Create environment
    print("  [1/8] 🏗️  Initializing environment...")
    env_config = EnvConfig(
        num_banks=num_banks,
        episode_length=num_steps,
        seed=42
    )
    env = FinancialEnvironment(env_config)
    observations, global_state = env.reset()
    print(f"        ✓ Network: {env.network.graph.number_of_edges()} edges")
    
    # Create exchanges
    print("\n  [2/8] 🏛️  Creating exchanges...")
    exchange_config = ExchangeConfig(
        max_throughput=1000,
        base_fee_rate=0.001,
        congestion_threshold=0.7,
        settlement_delay=1
    )
    
    exchanges = []
    for i in range(num_exchanges):
        exchange = Exchange(
            exchange_id=i,
            config=exchange_config
        )
        exchanges.append(exchange)
    
    exchange_network = ExchangeNetwork(exchanges)
    print(f"        ✓ Created {num_exchanges} exchanges")
    
    # Create CCPs
    print("\n  [3/8] 🏦 Creating CCPs (Clearing Houses)...")
    ccps = []
    for i in range(num_ccps):
        ccp = CentralCounterparty(
            ccp_id=i,
            initial_capital=10_000_000,
            initial_margin_rate=0.1,
            variation_margin_rate=0.05,
            default_fund_rate=0.02,
            netting_efficiency=0.6
        )
        # Register all banks as members
        for bank_id in range(num_banks):
            ccp.register_member(bank_id, initial_collateral=100_000)
        ccps.append(ccp)
    
    ccp_network = CCPNetwork(ccps)
    print(f"        ✓ Created {num_ccps} CCPs with {num_banks} members each")
    
    # Create infrastructure router
    print("\n  [4/8] 🔀 Creating infrastructure router...")
    router = InfrastructureRouter(
        exchanges=exchanges,
        ccps=ccps,
        banks=env.network.banks
    )
    print("        ✓ Router configured: Banks → Exchanges → CCPs → Banks")
    
    # Visualize infrastructure
    if visualize:
        print("\n  [5/8] 🗺️  Visualizing infrastructure network...")
        infra_viz.plot_infrastructure_network(
            exchanges=exchanges,
            ccps=ccps,
            banks=env.network.banks,
            router=router,
            save=save_plots,
            show=False
        )
    
    # Create agents
    print("\n  [6/8] 🤖 Creating agents...")
    agents = {}
    agent_types = ['rule_based', 'myopic', 'conservative', 'greedy']
    
    for i in range(num_banks):
        agent_type = agent_types[i % len(agent_types)]
        agent_config = AgentConfig(
            agent_id=i,
            observation_dim=FinancialEnvironment.OBS_DIM,
            action_dim=FinancialEnvironment.ACTION_DIM,
            seed=42 + i
        )
        agents[i] = create_baseline_agent(agent_type, agent_config)
    print(f"        ✓ Created {num_banks} agents")
    
    # Set scenario
    print(f"\n  [7/8] 📋 Setting scenario: {scenario}")
    scenario_engine = ScenarioEngine(seed=42)
    scenario_engine.set_scenario(scenario)
    
    # Run simulation
    print(f"\n  [8/8] 🚀 Running simulation with infrastructure routing...")
    print("  " + "─" * 66)
    
    total_rewards = {i: 0.0 for i in range(num_banks)}
    infrastructure_stats = {
        'total_transactions': 0,
        'exchange_fees': 0.0,
        'ccp_margins': 0.0,
        'settlement_delays': [],
        'congestion_levels': []
    }
    
    for step in range(num_steps):
        # Apply scenario shocks
        shocks = scenario_engine.generate_shocks(step, num_steps, num_banks)
        if any(v != 0 for v in shocks['market'].values()):
            env.apply_scenario_shock(**shocks['market'])
        
        # Get actions from all agents
        actions = {}
        for agent_id, agent in agents.items():
            obs = observations[agent_id]
            # Add infrastructure observation (8 dims)
            infra_obs = router.get_infrastructure_observation()
            # Extend observation with infrastructure state
            extended_obs = np.concatenate([obs, infra_obs])
            actions[agent_id] = agent.select_action(obs)  # Use original obs for action
        
        # Process actions through infrastructure
        for agent_id, action in actions.items():
            # Interpret action as trading decision
            lending_target = action[0]  # Lending adjustment
            liquidity_target = action[1]  # Liquidity buffer
            
            # Create transaction if lending
            if lending_target > 0.1:
                # Select random counterparty
                counterparty = np.random.choice([
                    i for i in range(num_banks) if i != agent_id
                ])
                amount = lending_target * 10000  # Scale
                
                # Route through infrastructure
                result = router.submit_transaction(
                    sender_id=agent_id,
                    receiver_id=counterparty,
                    amount=amount,
                    transaction_type='lending'
                )
                
                if result['success']:
                    infrastructure_stats['total_transactions'] += 1
                    infrastructure_stats['exchange_fees'] += result.get('fee', 0)
                    infrastructure_stats['settlement_delays'].append(
                        result.get('delay', 0)
                    )
        
        # Update exchange congestion
        for exchange in exchanges:
            exchange.update_congestion()
            infrastructure_stats['congestion_levels'].append(exchange.congestion)
        
        # Process CCP margin calls
        for ccp in ccps:
            ccp.process_margin_calls()
        
        # Step environment
        result = env.step(actions)
        
        # Accumulate rewards (adjusted for infrastructure costs)
        for agent_id, reward in result.rewards.items():
            # Deduct infrastructure fees
            infra_cost = infrastructure_stats['exchange_fees'] / max(num_banks, 1)
            adjusted_reward = reward - infra_cost * 0.01
            total_rewards[agent_id] += adjusted_reward
        
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
    
    # Infrastructure summary
    print("\n  🏛️  INFRASTRUCTURE SUMMARY")
    print("  " + "─" * 40)
    print(f"  │ Total Transactions:   {infrastructure_stats['total_transactions']:,}")
    print(f"  │ Exchange Fees:        ${infrastructure_stats['exchange_fees']:,.2f}")
    if infrastructure_stats['settlement_delays']:
        avg_delay = np.mean(infrastructure_stats['settlement_delays'])
        print(f"  │ Avg Settlement Delay: {avg_delay:.2f} steps")
    if infrastructure_stats['congestion_levels']:
        avg_congestion = np.mean(infrastructure_stats['congestion_levels'])
        max_congestion = max(infrastructure_stats['congestion_levels'])
        print(f"  │ Avg Congestion:       {avg_congestion:.2%}")
        print(f"  │ Max Congestion:       {max_congestion:.2%}")
    
    # Risk analysis
    print("\n  📊 Analyzing results...")
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
        
        # Simulation summary
        sim_viz.plot_simulation_summary(
            title=f"Infrastructure Simulation - {scenario.title()} Scenario",
            save=save_plots,
            show=False
        )
        
        # Transaction flow
        infra_viz.plot_transaction_flow(
            router=router,
            time_window=num_steps,
            save=save_plots,
            show=False
        )
        
        # CCP margin status
        if ccps:
            infra_viz.plot_margin_status(
                ccp=ccps[0],
                save=save_plots,
                show=False
            )
        
        # Network state
        net_viz.plot_network(
            env.network.graph,
            env.network.banks,
            title=f"Final Network - {scenario.title()} Scenario",
            save=save_plots,
            show=False
        )
        
        # Risk dashboard
        risk_dashboard.create_dashboard(
            report, final_stats, market_state,
            title=f"Risk Dashboard - {scenario.title()}",
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
        'risk_report': report,
        'infrastructure_stats': infrastructure_stats
    }


def main():
    parser = argparse.ArgumentParser(
        description="FinSim-MAPPO Infrastructure Demo"
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
        '--exchanges',
        type=int,
        default=2,
        help='Number of exchanges'
    )
    parser.add_argument(
        '--ccps',
        type=int,
        default=1,
        help='Number of CCPs'
    )
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Disable visualization'
    )
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Save plots but do not display'
    )
    
    args = parser.parse_args()
    
    run_infrastructure_demo(
        scenario=args.scenario,
        num_banks=args.banks,
        num_steps=args.steps,
        num_exchanges=args.exchanges,
        num_ccps=args.ccps,
        visualize=not args.no_viz,
        save_plots=True,
        show_plots=not args.no_show
    )


if __name__ == "__main__":
    main()
