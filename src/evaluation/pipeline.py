"""
Formal Evaluation Pipeline for FinSim-MAPPO.
Provides baseline comparisons, multi-seed validation, and comprehensive metrics.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics."""
    # Identification
    agent_type: str
    scenario: str
    seed: int
    
    # Performance
    total_reward: float = 0.0
    mean_reward: float = 0.0
    std_reward: float = 0.0
    
    # Stability
    default_rate: float = 0.0
    stressed_rate: float = 0.0
    max_defaults: int = 0
    time_to_first_default: int = -1
    
    # Liquidity
    avg_liquidity_ratio: float = 0.0
    min_liquidity_ratio: float = 0.0
    liquidity_crisis_steps: int = 0
    
    # Welfare (aggregate utility)
    total_welfare: float = 0.0
    gini_coefficient: float = 0.0  # Inequality in rewards
    
    # Recovery
    recovery_time: int = 0  # Steps to recover from crisis
    recovery_success: bool = True
    
    # Risk concentration
    herfindahl_index: float = 0.0  # Concentration of exposure
    max_exposure_ratio: float = 0.0
    
    # Episode info
    episode_length: int = 0
    early_termination: bool = False
    
    def to_dict(self) -> dict:
        return {
            'agent_type': self.agent_type,
            'scenario': self.scenario,
            'seed': self.seed,
            'total_reward': self.total_reward,
            'mean_reward': self.mean_reward,
            'std_reward': self.std_reward,
            'default_rate': self.default_rate,
            'stressed_rate': self.stressed_rate,
            'max_defaults': self.max_defaults,
            'time_to_first_default': self.time_to_first_default,
            'avg_liquidity_ratio': self.avg_liquidity_ratio,
            'min_liquidity_ratio': self.min_liquidity_ratio,
            'liquidity_crisis_steps': self.liquidity_crisis_steps,
            'total_welfare': self.total_welfare,
            'gini_coefficient': self.gini_coefficient,
            'recovery_time': self.recovery_time,
            'recovery_success': self.recovery_success,
            'herfindahl_index': self.herfindahl_index,
            'max_exposure_ratio': self.max_exposure_ratio,
            'episode_length': self.episode_length,
            'early_termination': self.early_termination
        }


@dataclass
class AggregatedResults:
    """Aggregated results across multiple seeds."""
    agent_type: str
    scenario: str
    num_seeds: int
    
    # Mean and std for each metric
    mean_reward: float = 0.0
    std_reward: float = 0.0
    mean_default_rate: float = 0.0
    std_default_rate: float = 0.0
    mean_welfare: float = 0.0
    std_welfare: float = 0.0
    mean_recovery_time: float = 0.0
    std_recovery_time: float = 0.0
    mean_liquidity: float = 0.0
    std_liquidity: float = 0.0
    
    # Success rates
    recovery_success_rate: float = 0.0
    zero_default_rate: float = 0.0
    
    # Individual results
    individual_results: List[EvaluationMetrics] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            'agent_type': self.agent_type,
            'scenario': self.scenario,
            'num_seeds': self.num_seeds,
            'mean_reward': self.mean_reward,
            'std_reward': self.std_reward,
            'mean_default_rate': self.mean_default_rate,
            'std_default_rate': self.std_default_rate,
            'mean_welfare': self.mean_welfare,
            'std_welfare': self.std_welfare,
            'mean_recovery_time': self.mean_recovery_time,
            'std_recovery_time': self.std_recovery_time,
            'mean_liquidity': self.mean_liquidity,
            'std_liquidity': self.std_liquidity,
            'recovery_success_rate': self.recovery_success_rate,
            'zero_default_rate': self.zero_default_rate
        }


class EvaluationPipeline:
    """
    Formal evaluation pipeline for comparing agent performance.
    
    Features:
    - Baseline comparison (rule-based, myopic, random)
    - Multi-seed validation
    - Comprehensive performance metrics
    """
    
    def __init__(self, 
                 env_config: Any,
                 scenarios: List[str] = None,
                 seeds: List[int] = None,
                 episode_length: int = 100,
                 output_dir: str = "outputs/evaluation"):
        
        self.env_config = env_config
        self.scenarios = scenarios or ['normal', 'liquidity_crisis', 'asset_crash', 'systemic']
        self.seeds = seeds or [42, 123, 456, 789, 1024]
        self.episode_length = episode_length
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: Dict[str, Dict[str, AggregatedResults]] = {}
    
    def evaluate_agent(self, 
                       agent_factory: callable,
                       agent_type: str,
                       scenario: str,
                       seed: int) -> EvaluationMetrics:
        """Evaluate a single agent on a single scenario with a single seed."""
        # Import here to avoid circular imports
        from src.environment import FinancialEnvironment, EnvConfig
        from src.scenarios import ScenarioEngine
        
        # Set seed
        np.random.seed(seed)
        
        # Create environment
        env_config = EnvConfig(
            num_banks=self.env_config.get('num_banks', 20),
            episode_length=self.episode_length,
            seed=seed
        )
        env = FinancialEnvironment(env_config)
        
        # Create agents
        agents = {}
        for i in range(env.num_agents):
            agents[i] = agent_factory(i, env.OBS_DIM, env.ACTION_DIM, seed + i)
        
        # Set scenario
        scenario_engine = ScenarioEngine(seed=seed)
        scenario_engine.set_scenario(scenario)
        
        # Initialize tracking
        observations, _ = env.reset()
        total_rewards = {i: 0.0 for i in range(env.num_agents)}
        
        default_history = []
        liquidity_history = []
        exposure_history = []
        first_default_step = -1
        crisis_start = -1
        recovery_step = -1
        
        # Run episode
        for step in range(self.episode_length):
            # Apply shocks
            shocks = scenario_engine.generate_shocks(step, self.episode_length, env.num_agents)
            if shocks['market']:
                env.apply_scenario_shock(**shocks['market'])
            
            # Get actions
            actions = {}
            for agent_id, agent in agents.items():
                actions[agent_id] = agent.select_action(observations[agent_id])
            
            # Step
            result = env.step(actions)
            
            # Track metrics
            for agent_id, reward in result.rewards.items():
                total_rewards[agent_id] += reward
            
            stats = result.network_stats
            default_history.append(stats.num_defaulted)
            
            # Track first default
            if stats.num_defaulted > 0 and first_default_step < 0:
                first_default_step = step
            
            # Track liquidity (simplified)
            avg_liquidity = np.mean([
                b.balance_sheet.cash / max(b.balance_sheet.total_liabilities, 1)
                for b in env.network.banks.values()
            ])
            liquidity_history.append(avg_liquidity)
            
            # Track exposure
            exposure_history.append(stats.total_exposure)
            
            # Track crisis and recovery
            if stats.num_defaulted > env.num_agents * 0.1 and crisis_start < 0:
                crisis_start = step
            if crisis_start >= 0 and stats.num_defaulted < env.num_agents * 0.05 and recovery_step < 0:
                recovery_step = step
            
            observations = result.observations
            
            if all(result.dones.values()):
                break
        
        # Calculate metrics
        rewards = list(total_rewards.values())
        final_stats = env.network.get_network_stats()
        
        # Gini coefficient
        sorted_rewards = np.sort(rewards)
        n = len(sorted_rewards)
        cumulative = np.cumsum(sorted_rewards)
        gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n if cumulative[-1] > 0 else 0
        
        # Herfindahl index (concentration)
        exposures = [sum(env.network.liability_matrix[i]) for i in range(env.num_agents)]
        total_exposure = sum(exposures)
        if total_exposure > 0:
            shares = [e / total_exposure for e in exposures]
            hhi = sum(s ** 2 for s in shares)
        else:
            hhi = 0
        
        return EvaluationMetrics(
            agent_type=agent_type,
            scenario=scenario,
            seed=seed,
            total_reward=sum(rewards),
            mean_reward=np.mean(rewards),
            std_reward=np.std(rewards),
            default_rate=final_stats.num_defaulted / env.num_agents,
            stressed_rate=final_stats.num_stressed / env.num_agents,
            max_defaults=max(default_history),
            time_to_first_default=first_default_step,
            avg_liquidity_ratio=np.mean(liquidity_history),
            min_liquidity_ratio=min(liquidity_history),
            liquidity_crisis_steps=sum(1 for l in liquidity_history if l < 0.1),
            total_welfare=sum(rewards),  # Simplified welfare = total rewards
            gini_coefficient=gini,
            recovery_time=recovery_step - crisis_start if recovery_step >= 0 else -1,
            recovery_success=recovery_step >= 0 or crisis_start < 0,
            herfindahl_index=hhi,
            max_exposure_ratio=max(exposures) / max(total_exposure, 1),
            episode_length=env.current_step,
            early_termination=env.current_step < self.episode_length
        )
    
    def evaluate_multi_seed(self,
                            agent_factory: callable,
                            agent_type: str,
                            scenario: str,
                            show_progress: bool = True) -> AggregatedResults:
        """Evaluate agent across multiple seeds."""
        results = []
        
        iterator = tqdm(self.seeds, desc=f"{agent_type} on {scenario}") if show_progress else self.seeds
        
        for seed in iterator:
            metrics = self.evaluate_agent(agent_factory, agent_type, scenario, seed)
            results.append(metrics)
        
        # Aggregate
        aggregated = AggregatedResults(
            agent_type=agent_type,
            scenario=scenario,
            num_seeds=len(self.seeds),
            mean_reward=np.mean([r.mean_reward for r in results]),
            std_reward=np.std([r.mean_reward for r in results]),
            mean_default_rate=np.mean([r.default_rate for r in results]),
            std_default_rate=np.std([r.default_rate for r in results]),
            mean_welfare=np.mean([r.total_welfare for r in results]),
            std_welfare=np.std([r.total_welfare for r in results]),
            mean_recovery_time=np.mean([r.recovery_time for r in results if r.recovery_time >= 0]),
            std_recovery_time=np.std([r.recovery_time for r in results if r.recovery_time >= 0]) if any(r.recovery_time >= 0 for r in results) else 0,
            mean_liquidity=np.mean([r.avg_liquidity_ratio for r in results]),
            std_liquidity=np.std([r.avg_liquidity_ratio for r in results]),
            recovery_success_rate=np.mean([r.recovery_success for r in results]),
            zero_default_rate=np.mean([r.default_rate == 0 for r in results]),
            individual_results=results
        )
        
        return aggregated
    
    def run_baseline_comparison(self, 
                                 trained_agent_factory: Optional[callable] = None,
                                 show_progress: bool = True) -> Dict[str, Dict[str, AggregatedResults]]:
        """
        Run comparison against baseline agents.
        
        Baselines:
        - Random agent
        - Rule-based agent
        - Myopic agent
        """
        from src.agents.baseline_agents import create_baseline_agent, AgentConfig
        
        def make_factory(agent_type: str) -> callable:
            def factory(agent_id: int, obs_dim: int, action_dim: int, seed: int):
                config = AgentConfig(agent_id=agent_id, observation_dim=obs_dim, 
                                    action_dim=action_dim, seed=seed)
                return create_baseline_agent(agent_type, config)
            return factory
        
        # Define agents to evaluate
        agent_factories = {
            'random': make_factory('random'),
            'rule_based': make_factory('rule_based'),
            'myopic': make_factory('myopic'),
            'greedy': make_factory('greedy'),
            'conservative': make_factory('conservative')
        }
        
        if trained_agent_factory:
            agent_factories['trained_mappo'] = trained_agent_factory
        
        # Run evaluations
        for agent_type, factory in agent_factories.items():
            self.results[agent_type] = {}
            
            for scenario in self.scenarios:
                aggregated = self.evaluate_multi_seed(factory, agent_type, scenario, show_progress)
                self.results[agent_type][scenario] = aggregated
        
        return self.results
    
    def generate_report(self) -> str:
        """Generate a formatted evaluation report."""
        lines = []
        lines.append("=" * 80)
        lines.append("FINSIM-MAPPO EVALUATION REPORT")
        lines.append(f"Generated: {datetime.now().isoformat()}")
        lines.append("=" * 80)
        lines.append("")
        
        for scenario in self.scenarios:
            lines.append(f"\n{'='*40}")
            lines.append(f"SCENARIO: {scenario.upper()}")
            lines.append(f"{'='*40}")
            
            # Header
            lines.append(f"\n{'Agent':<15} {'Mean Reward':>12} {'Default Rate':>12} "
                        f"{'Welfare':>12} {'Liquidity':>10}")
            lines.append("-" * 65)
            
            # Results for each agent
            for agent_type in self.results:
                if scenario in self.results[agent_type]:
                    r = self.results[agent_type][scenario]
                    lines.append(
                        f"{agent_type:<15} "
                        f"{r.mean_reward:>10.2f}±{r.std_reward:.1f} "
                        f"{r.mean_default_rate*100:>10.1f}%±{r.std_default_rate*100:.1f} "
                        f"{r.mean_welfare:>10.1f}±{r.std_welfare:.1f} "
                        f"{r.mean_liquidity:>8.2f}±{r.std_liquidity:.2f}"
                    )
            
            lines.append("")
        
        # Summary statistics
        lines.append("\n" + "=" * 80)
        lines.append("SUMMARY ACROSS ALL SCENARIOS")
        lines.append("=" * 80)
        
        for agent_type in self.results:
            all_rewards = []
            all_defaults = []
            
            for scenario in self.results[agent_type]:
                r = self.results[agent_type][scenario]
                all_rewards.extend([ir.mean_reward for ir in r.individual_results])
                all_defaults.extend([ir.default_rate for ir in r.individual_results])
            
            lines.append(f"\n{agent_type}:")
            lines.append(f"  Overall Mean Reward: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
            lines.append(f"  Overall Default Rate: {np.mean(all_defaults)*100:.1f}% ± {np.std(all_defaults)*100:.1f}%")
        
        return "\n".join(lines)
    
    def save_results(self) -> None:
        """Save evaluation results to files."""
        # Save JSON results
        json_results = {}
        for agent_type in self.results:
            json_results[agent_type] = {}
            for scenario in self.results[agent_type]:
                json_results[agent_type][scenario] = self.results[agent_type][scenario].to_dict()
        
        json_path = self.output_dir / "evaluation_results.json"
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Save report
        report = self.generate_report()
        report_path = self.output_dir / "evaluation_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Results saved to {self.output_dir}")
    
    def plot_comparison(self, save: bool = True, show: bool = True):
        """Generate comparison plots."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        agent_types = list(self.results.keys())
        x = np.arange(len(self.scenarios))
        width = 0.8 / len(agent_types)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(agent_types)))
        
        # Plot 1: Mean Reward
        ax1 = axes[0, 0]
        for i, agent_type in enumerate(agent_types):
            rewards = [self.results[agent_type][s].mean_reward for s in self.scenarios]
            stds = [self.results[agent_type][s].std_reward for s in self.scenarios]
            ax1.bar(x + i * width, rewards, width, label=agent_type, color=colors[i], 
                    yerr=stds, capsize=2)
        ax1.set_title('Mean Reward by Scenario')
        ax1.set_xticks(x + width * len(agent_types) / 2)
        ax1.set_xticklabels([s.replace('_', '\n') for s in self.scenarios])
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Default Rate
        ax2 = axes[0, 1]
        for i, agent_type in enumerate(agent_types):
            defaults = [self.results[agent_type][s].mean_default_rate * 100 for s in self.scenarios]
            stds = [self.results[agent_type][s].std_default_rate * 100 for s in self.scenarios]
            ax2.bar(x + i * width, defaults, width, label=agent_type, color=colors[i],
                    yerr=stds, capsize=2)
        ax2.set_title('Default Rate (%) by Scenario')
        ax2.set_xticks(x + width * len(agent_types) / 2)
        ax2.set_xticklabels([s.replace('_', '\n') for s in self.scenarios])
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Welfare
        ax3 = axes[1, 0]
        for i, agent_type in enumerate(agent_types):
            welfare = [self.results[agent_type][s].mean_welfare for s in self.scenarios]
            ax3.bar(x + i * width, welfare, width, label=agent_type, color=colors[i])
        ax3.set_title('Total Welfare by Scenario')
        ax3.set_xticks(x + width * len(agent_types) / 2)
        ax3.set_xticklabels([s.replace('_', '\n') for s in self.scenarios])
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Recovery Success Rate
        ax4 = axes[1, 1]
        for i, agent_type in enumerate(agent_types):
            recovery = [self.results[agent_type][s].recovery_success_rate * 100 for s in self.scenarios]
            ax4.bar(x + i * width, recovery, width, label=agent_type, color=colors[i])
        ax4.set_title('Recovery Success Rate (%) by Scenario')
        ax4.set_xticks(x + width * len(agent_types) / 2)
        ax4.set_xticklabels([s.replace('_', '\n') for s in self.scenarios])
        ax4.set_ylim(0, 105)
        ax4.grid(True, alpha=0.3, axis='y')
        
        fig.suptitle('Agent Comparison Across Scenarios', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / "comparison_plot.png", dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close(fig)


def run_quick_evaluation(num_seeds: int = 3, episode_length: int = 50):
    """Run a quick evaluation for testing."""
    from src.environment import EnvConfig
    
    config = {'num_banks': 15}
    seeds = list(range(42, 42 + num_seeds))
    
    pipeline = EvaluationPipeline(
        env_config=config,
        scenarios=['normal', 'liquidity_crisis'],
        seeds=seeds,
        episode_length=episode_length
    )
    
    results = pipeline.run_baseline_comparison(show_progress=True)
    
    print(pipeline.generate_report())
    pipeline.save_results()
    pipeline.plot_comparison(save=True, show=True)
    
    return results
