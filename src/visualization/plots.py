"""
Visualization Module for FinSim-MAPPO
Provides plotting and visualization utilities for understanding simulation dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import networkx as nx
from pathlib import Path
import json
from datetime import datetime


@dataclass
class SimulationHistory:
    """Stores history of simulation for visualization."""
    steps: List[int] = field(default_factory=list)
    asset_prices: List[float] = field(default_factory=list)
    default_counts: List[int] = field(default_factory=list)
    stressed_counts: List[int] = field(default_factory=list)
    avg_capital_ratios: List[float] = field(default_factory=list)
    total_rewards: List[float] = field(default_factory=list)
    liquidity_indices: List[float] = field(default_factory=list)
    volatilities: List[float] = field(default_factory=list)
    total_lending: List[float] = field(default_factory=list)
    bank_equities: Dict[int, List[float]] = field(default_factory=dict)
    
    def record(self, step: int, market_state, network_stats, rewards: Dict[int, float],
               bank_equities: Optional[Dict[int, float]] = None):
        """Record a simulation step."""
        self.steps.append(step)
        self.asset_prices.append(market_state.asset_price)
        self.default_counts.append(network_stats.num_defaulted)
        self.stressed_counts.append(network_stats.num_stressed)
        self.avg_capital_ratios.append(network_stats.avg_capital_ratio)
        self.total_rewards.append(sum(rewards.values()))
        self.liquidity_indices.append(getattr(market_state, 'liquidity_index', 1.0))
        self.volatilities.append(getattr(market_state, 'volatility', 0.02))
        self.total_lending.append(network_stats.total_exposure)
        
        if bank_equities:
            for bank_id, equity in bank_equities.items():
                if bank_id not in self.bank_equities:
                    self.bank_equities[bank_id] = []
                self.bank_equities[bank_id].append(equity)


@dataclass 
class TrainingHistory:
    """Stores training metrics history."""
    episodes: List[int] = field(default_factory=list)
    mean_rewards: List[float] = field(default_factory=list)
    default_rates: List[float] = field(default_factory=list)
    actor_losses: List[float] = field(default_factory=list)
    critic_losses: List[float] = field(default_factory=list)
    entropy: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    curriculum_stage: List[int] = field(default_factory=list)
    
    def record(self, episode: int, mean_reward: float, default_rate: float,
               actor_loss: float = 0, critic_loss: float = 0, 
               entropy: float = 0, episode_length: int = 100,
               stage: int = 0):
        """Record training episode metrics."""
        self.episodes.append(episode)
        self.mean_rewards.append(mean_reward)
        self.default_rates.append(default_rate)
        self.actor_losses.append(actor_loss)
        self.critic_losses.append(critic_loss)
        self.entropy.append(entropy)
        self.episode_lengths.append(episode_length)
        self.curriculum_stage.append(stage)


class SimulationVisualizer:
    """Visualize simulation dynamics in real-time or post-hoc."""
    
    def __init__(self, save_dir: str = "outputs/figures"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.history = SimulationHistory()
        
    def record_step(self, step: int, market_state, network_stats, 
                    rewards: Dict[int, float], bank_equities: Optional[Dict[int, float]] = None):
        """Record a simulation step for later visualization."""
        self.history.record(step, market_state, network_stats, rewards, bank_equities)
    
    def plot_simulation_summary(self, title: str = "Simulation Summary", 
                                 save: bool = True, show: bool = True) -> plt.Figure:
        """Create a comprehensive summary plot of the simulation."""
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Color scheme
        colors = {
            'price': '#2ecc71',
            'defaults': '#e74c3c', 
            'stressed': '#f39c12',
            'capital': '#3498db',
            'reward': '#9b59b6',
            'liquidity': '#1abc9c'
        }
        
        # 1. Asset Price Evolution
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(self.history.steps, self.history.asset_prices, 
                 color=colors['price'], linewidth=2)
        ax1.fill_between(self.history.steps, self.history.asset_prices, 
                         alpha=0.3, color=colors['price'])
        ax1.set_title('Asset Price Evolution', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Price')
        ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Initial Price')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # 2. Bank Status (Defaults & Stressed)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.stackplot(self.history.steps, 
                      self.history.default_counts, 
                      self.history.stressed_counts,
                      labels=['Defaulted', 'Stressed'],
                      colors=[colors['defaults'], colors['stressed']],
                      alpha=0.7)
        ax2.set_title('Bank Status Over Time', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Number of Banks')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # 3. Capital Ratio
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(self.history.steps, 
                 [cr * 100 for cr in self.history.avg_capital_ratios],
                 color=colors['capital'], linewidth=2)
        ax3.axhline(y=8, color='red', linestyle='--', alpha=0.7, label='Min Requirement (8%)')
        ax3.fill_between(self.history.steps, 
                         [cr * 100 for cr in self.history.avg_capital_ratios],
                         alpha=0.3, color=colors['capital'])
        ax3.set_title('Average Capital Ratio', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Capital Ratio (%)')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        
        # 4. Cumulative Rewards
        ax4 = fig.add_subplot(gs[1, 0])
        cumulative_rewards = np.cumsum(self.history.total_rewards)
        ax4.plot(self.history.steps, cumulative_rewards, 
                 color=colors['reward'], linewidth=2)
        ax4.fill_between(self.history.steps, cumulative_rewards, 
                         alpha=0.3, color=colors['reward'])
        ax4.set_title('Cumulative System Reward', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Total Reward')
        ax4.grid(True, alpha=0.3)
        
        # 5. Step Rewards
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.bar(self.history.steps, self.history.total_rewards, 
                color=colors['reward'], alpha=0.7, width=1.0)
        ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax5.set_title('Step-wise Total Reward', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Step')
        ax5.set_ylabel('Reward')
        ax5.grid(True, alpha=0.3)
        
        # 6. Volatility
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.plot(self.history.steps, self.history.volatilities, 
                 color='#e67e22', linewidth=2)
        ax6.fill_between(self.history.steps, self.history.volatilities, 
                         alpha=0.3, color='#e67e22')
        ax6.set_title('Market Volatility', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Step')
        ax6.set_ylabel('Volatility')
        ax6.grid(True, alpha=0.3)
        
        # 7. Total Interbank Exposure
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.plot(self.history.steps, self.history.total_lending, 
                 color='#34495e', linewidth=2)
        ax7.fill_between(self.history.steps, self.history.total_lending, 
                         alpha=0.3, color='#34495e')
        ax7.set_title('Total Interbank Exposure', fontsize=12, fontweight='bold')
        ax7.set_xlabel('Step')
        ax7.set_ylabel('Exposure')
        ax7.grid(True, alpha=0.3)
        
        # 8. Summary Statistics Box
        ax8 = fig.add_subplot(gs[2, 1:])
        ax8.axis('off')
        
        # Calculate summary stats
        final_defaults = self.history.default_counts[-1] if self.history.default_counts else 0
        max_defaults = max(self.history.default_counts) if self.history.default_counts else 0
        final_price = self.history.asset_prices[-1] if self.history.asset_prices else 1.0
        min_price = min(self.history.asset_prices) if self.history.asset_prices else 1.0
        total_reward = sum(self.history.total_rewards)
        avg_cr = np.mean(self.history.avg_capital_ratios) if self.history.avg_capital_ratios else 0
        
        summary_text = f"""
╔══════════════════════════════════════════════════════════════════╗
║                     SIMULATION SUMMARY                           ║
╠══════════════════════════════════════════════════════════════════╣
║  Total Steps:        {len(self.history.steps):>6}                                    ║
║  Final Defaults:     {final_defaults:>6} banks     (Max: {max_defaults})                      ║
║  Final Asset Price:  {final_price:>6.3f}         (Min: {min_price:.3f})                     ║
║  Avg Capital Ratio:  {avg_cr*100:>6.2f}%                                       ║
║  Total Reward:       {total_reward:>10.2f}                                   ║
╚══════════════════════════════════════════════════════════════════╝
        """
        ax8.text(0.5, 0.5, summary_text, transform=ax8.transAxes,
                fontsize=11, fontfamily='monospace',
                verticalalignment='center', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.save_dir / f"simulation_summary_{timestamp}.png"
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"    📊 Saved simulation summary to {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
            
        return fig
    
    def plot_bank_trajectories(self, num_banks: int = 10, 
                                save: bool = True, show: bool = True) -> plt.Figure:
        """Plot equity trajectories for individual banks."""
        if not self.history.bank_equities:
            print("No bank equity data recorded.")
            return None
            
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Select banks to plot
        bank_ids = list(self.history.bank_equities.keys())[:num_banks]
        cmap = plt.cm.get_cmap('tab20', len(bank_ids))
        
        for i, bank_id in enumerate(bank_ids):
            equities = self.history.bank_equities[bank_id]
            steps = list(range(len(equities)))
            ax.plot(steps, equities, color=cmap(i), linewidth=1.5, 
                    alpha=0.7, label=f'Bank {bank_id}')
        
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Default Threshold')
        ax.set_title('Individual Bank Equity Trajectories', fontsize=14, fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Equity')
        ax.legend(loc='upper right', ncol=2, fontsize=8)
        ax.grid(True, alpha=0.3)
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.save_dir / f"bank_trajectories_{timestamp}.png"
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"    📊 Saved bank trajectories to {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
            
        return fig


class TrainingVisualizer:
    """Visualize training progress and metrics."""
    
    def __init__(self, save_dir: str = "outputs/figures"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.history = TrainingHistory()
        
    def record_episode(self, episode: int, mean_reward: float, default_rate: float,
                       actor_loss: float = 0, critic_loss: float = 0,
                       entropy: float = 0, episode_length: int = 100, stage: int = 0):
        """Record training episode metrics."""
        self.history.record(episode, mean_reward, default_rate, 
                           actor_loss, critic_loss, entropy, episode_length, stage)
    
    def plot_training_curves(self, window: int = 50, 
                              save: bool = True, show: bool = True) -> plt.Figure:
        """Plot comprehensive training curves."""
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        episodes = self.history.episodes
        
        def smooth(values, window):
            if len(values) < window:
                return values
            return np.convolve(values, np.ones(window)/window, mode='valid')
        
        # 1. Mean Reward
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(episodes, self.history.mean_rewards, alpha=0.3, color='#3498db')
        smoothed = smooth(self.history.mean_rewards, window)
        smooth_x = episodes[window-1:] if len(episodes) >= window else episodes
        if len(smoothed) == len(smooth_x):
            ax1.plot(smooth_x, smoothed, color='#3498db', linewidth=2, label=f'{window}-ep avg')
        ax1.set_title('Mean Episode Reward', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Default Rate
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(episodes, [dr * 100 for dr in self.history.default_rates], 
                 alpha=0.3, color='#e74c3c')
        smoothed = smooth([dr * 100 for dr in self.history.default_rates], window)
        if len(smoothed) == len(smooth_x):
            ax2.plot(smooth_x, smoothed, color='#e74c3c', linewidth=2)
        ax2.set_title('Default Rate (%)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Default Rate (%)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Episode Length
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(episodes, self.history.episode_lengths, alpha=0.3, color='#2ecc71')
        smoothed = smooth(self.history.episode_lengths, window)
        if len(smoothed) == len(smooth_x):
            ax3.plot(smooth_x, smoothed, color='#2ecc71', linewidth=2)
        ax3.set_title('Episode Length', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Steps')
        ax3.grid(True, alpha=0.3)
        
        # 4. Actor Loss
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(episodes, self.history.actor_losses, alpha=0.3, color='#9b59b6')
        smoothed = smooth(self.history.actor_losses, window)
        if len(smoothed) == len(smooth_x):
            ax4.plot(smooth_x, smoothed, color='#9b59b6', linewidth=2)
        ax4.set_title('Actor Loss', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Loss')
        ax4.grid(True, alpha=0.3)
        
        # 5. Critic Loss  
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(episodes, self.history.critic_losses, alpha=0.3, color='#f39c12')
        smoothed = smooth(self.history.critic_losses, window)
        if len(smoothed) == len(smooth_x):
            ax5.plot(smooth_x, smoothed, color='#f39c12', linewidth=2)
        ax5.set_title('Critic Loss', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Episode')
        ax5.set_ylabel('Loss')
        ax5.grid(True, alpha=0.3)
        
        # 6. Curriculum Stage
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.step(episodes, self.history.curriculum_stage, where='post', 
                 color='#1abc9c', linewidth=2)
        ax6.set_title('Curriculum Stage', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Episode')
        ax6.set_ylabel('Stage')
        ax6.set_ylim(-0.5, max(self.history.curriculum_stage) + 0.5 if self.history.curriculum_stage else 3.5)
        ax6.grid(True, alpha=0.3)
        
        fig.suptitle('Training Progress', fontsize=14, fontweight='bold', y=0.98)
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.save_dir / f"training_curves_{timestamp}.png"
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"📊 Saved training curves to {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
            
        return fig
    
    def print_progress(self, episode: int, total_episodes: int, 
                       reward: float, default_rate: float, 
                       actor_loss: float = 0, critic_loss: float = 0,
                       stage: int = 0):
        """Print formatted training progress."""
        bar_length = 30
        progress = episode / total_episodes
        filled = int(bar_length * progress)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        print(f"\r[{bar}] {episode:4d}/{total_episodes} | "
              f"R: {reward:7.1f} | "
              f"Def: {default_rate*100:5.1f}% | "
              f"πL: {actor_loss:6.3f} | "
              f"VL: {critic_loss:6.3f} | "
              f"Stage: {stage}", end='', flush=True)


class NetworkVisualizer:
    """Visualize financial network topology and dynamics."""
    
    def __init__(self, save_dir: str = "outputs/figures"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_network(self, graph: nx.DiGraph, banks: dict,
                     title: str = "Financial Network",
                     save: bool = True, show: bool = True) -> plt.Figure:
        """Plot the financial network with bank status coloring."""
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Node colors based on status
        node_colors = []
        node_sizes = []
        
        for node in graph.nodes():
            bank = banks.get(node)
            if bank is None:
                node_colors.append('gray')
                node_sizes.append(300)
            else:
                # Color based on status
                status = bank.status.value if hasattr(bank.status, 'value') else str(bank.status)
                if 'DEFAULTED' in status.upper():
                    node_colors.append('#e74c3c')  # Red
                elif 'STRESSED' in status.upper():
                    node_colors.append('#f39c12')  # Orange
                else:
                    node_colors.append('#2ecc71')  # Green
                
                # Size based on tier (core vs periphery)
                tier = getattr(bank, 'tier', 2)
                node_sizes.append(600 if tier == 1 else 300)
        
        # Layout
        pos = nx.spring_layout(graph, k=2, iterations=50, seed=42)
        
        # Draw edges with varying width based on exposure
        edges = graph.edges(data=True)
        edge_weights = [d.get('weight', 1) for u, v, d in edges]
        max_weight = max(edge_weights) if edge_weights else 1
        edge_widths = [0.5 + 2 * w / max_weight for w in edge_weights]
        
        nx.draw_networkx_edges(graph, pos, ax=ax, 
                               width=edge_widths,
                               alpha=0.4,
                               edge_color='gray',
                               arrows=True,
                               arrowsize=10)
        
        # Draw nodes
        nx.draw_networkx_nodes(graph, pos, ax=ax,
                               node_color=node_colors,
                               node_size=node_sizes,
                               alpha=0.9,
                               edgecolors='black',
                               linewidths=1)
        
        # Labels
        nx.draw_networkx_labels(graph, pos, ax=ax,
                                font_size=8,
                                font_weight='bold')
        
        # Legend
        legend_elements = [
            mpatches.Patch(color='#2ecc71', label='Active'),
            mpatches.Patch(color='#f39c12', label='Stressed'),
            mpatches.Patch(color='#e74c3c', label='Defaulted'),
            plt.Line2D([0], [0], marker='o', color='w', 
                       markerfacecolor='gray', markersize=15, label='Core Bank'),
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor='gray', markersize=10, label='Periphery Bank')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.save_dir / f"network_{timestamp}.png"
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"    📊 Saved network visualization to {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
            
        return fig
    
    def plot_exposure_heatmap(self, liability_matrix: np.ndarray,
                               title: str = "Interbank Exposure Matrix",
                               save: bool = True, show: bool = True) -> plt.Figure:
        """Plot heatmap of interbank exposures."""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap
        im = ax.imshow(liability_matrix, cmap='YlOrRd', aspect='auto')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Exposure Amount', rotation=270, labelpad=20)
        
        # Labels
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Creditor Bank')
        ax.set_ylabel('Debtor Bank')
        
        # Add text annotations for larger matrices (only if small enough)
        n = liability_matrix.shape[0]
        if n <= 15:
            for i in range(n):
                for j in range(n):
                    val = liability_matrix[i, j]
                    if val > 0:
                        ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                                fontsize=7, color='black' if val < liability_matrix.max()/2 else 'white')
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.save_dir / f"exposure_heatmap_{timestamp}.png"
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"    📊 Saved exposure heatmap to {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
            
        return fig


class RiskDashboard:
    """Create comprehensive risk dashboard visualization."""
    
    def __init__(self, save_dir: str = "outputs/figures"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def create_dashboard(self, risk_report, network_stats, market_state,
                         title: str = "Risk Dashboard",
                         save: bool = True, show: bool = True) -> plt.Figure:
        """Create a comprehensive risk dashboard."""
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.3)
        
        # 1. Risk Gauges (top row)
        # DebtRank gauge
        ax1 = fig.add_subplot(gs[0, 0])
        self._draw_gauge(ax1, risk_report.debt_rank, "DebtRank", 
                         thresholds=[0.3, 0.6], colors=['#2ecc71', '#f39c12', '#e74c3c'])
        
        # Systemic Risk Index gauge
        ax2 = fig.add_subplot(gs[0, 1])
        self._draw_gauge(ax2, risk_report.systemic_risk_index, "Systemic Risk",
                         thresholds=[0.3, 0.6], colors=['#2ecc71', '#f39c12', '#e74c3c'])
        
        # Liquidity Index gauge
        ax3 = fig.add_subplot(gs[0, 2])
        self._draw_gauge(ax3, risk_report.liquidity_index, "Liquidity",
                         thresholds=[0.3, 0.6], colors=['#e74c3c', '#f39c12', '#2ecc71'], invert=True)
        
        # Stress Index gauge
        ax4 = fig.add_subplot(gs[0, 3])
        self._draw_gauge(ax4, risk_report.stress_index, "Stress Index",
                         thresholds=[0.3, 0.6], colors=['#2ecc71', '#f39c12', '#e74c3c'])
        
        # 2. Network Status (middle left)
        ax5 = fig.add_subplot(gs[1, :2])
        statuses = ['Active', 'Stressed', 'Defaulted']
        counts = [
            network_stats.num_banks - network_stats.num_stressed - network_stats.num_defaulted,
            network_stats.num_stressed,
            network_stats.num_defaulted
        ]
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        
        bars = ax5.bar(statuses, counts, color=colors, edgecolor='black')
        ax5.set_title('Bank Status Distribution', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Number of Banks')
        
        for bar, count in zip(bars, counts):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     str(count), ha='center', fontsize=12, fontweight='bold')
        
        # 3. Market Status (middle right)
        ax6 = fig.add_subplot(gs[1, 2:])
        metrics = ['Asset Price', 'Volatility', 'Liquidity Index', 'Avg Cap Ratio']
        values = [
            market_state.asset_price,
            market_state.volatility * 100,  # Convert to percentage
            market_state.liquidity_index,
            network_stats.avg_capital_ratio * 100  # Convert to percentage
        ]
        targets = [1.0, 2.0, 1.0, 8.0]  # Target/baseline values
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax6.bar(x - width/2, values, width, label='Current', color='#3498db')
        bars2 = ax6.bar(x + width/2, targets, width, label='Target/Baseline', 
                        color='#95a5a6', alpha=0.7)
        
        ax6.set_title('Key Metrics vs Targets', fontsize=12, fontweight='bold')
        ax6.set_xticks(x)
        ax6.set_xticklabels(metrics, rotation=15)
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')
        
        # 4. Risk Summary (bottom)
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        
        # Build summary text
        health = self._assess_health(risk_report)
        
        summary = f"""
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              SYSTEM RISK ASSESSMENT                                     │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  Overall Health:  {health['status']:12s}    Cascade Potential: {risk_report.cascade_potential*100:5.1f}%                         │
│                                                                                         │
│  Contagion Depth: {risk_report.contagion_depth:3d} layers         Risk Grade: {health['grade']:1s}                                      │
│                                                                                         │
│  Systemically Important Banks: {str(risk_report.systemically_important_banks):<45s}    │
│  Vulnerable Banks:             {str(risk_report.vulnerable_banks):<45s}    │
│                                                                                         │
│  {health['message']:<85s} │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
        """
        
        ax7.text(0.5, 0.5, summary, transform=ax7.transAxes,
                 fontsize=10, fontfamily='monospace',
                 verticalalignment='center', horizontalalignment='center',
                 bbox=dict(boxstyle='round', facecolor=health['color'], alpha=0.3))
        
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.save_dir / f"risk_dashboard_{timestamp}.png"
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"    📊 Saved risk dashboard to {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
            
        return fig
    
    def _draw_gauge(self, ax, value: float, title: str, 
                    thresholds: List[float] = [0.3, 0.6],
                    colors: List[str] = ['green', 'yellow', 'red'],
                    invert: bool = False):
        """Draw a simple gauge chart."""
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Clamp value
        value = max(0, min(1, value))
        
        # Draw arc segments
        import matplotlib.patches as patches
        
        # Background arc
        theta1, theta2 = 180, 0
        arc = patches.Arc((0, 0), 2, 2, angle=0, theta1=theta1, theta2=theta2,
                          linewidth=20, color='#ecf0f1')
        ax.add_patch(arc)
        
        # Colored segments
        segment_angles = [180, 180 - thresholds[0] * 180, 
                          180 - thresholds[1] * 180, 0]
        for i in range(3):
            arc = patches.Arc((0, 0), 2, 2, angle=0, 
                             theta1=segment_angles[i+1], theta2=segment_angles[i],
                             linewidth=18, color=colors[i] if not invert else colors[2-i])
            ax.add_patch(arc)
        
        # Needle
        angle = np.radians(180 - value * 180)
        needle_x = 0.9 * np.cos(angle)
        needle_y = 0.9 * np.sin(angle)
        ax.arrow(0, 0, needle_x, needle_y, head_width=0.08, head_length=0.05,
                 fc='black', ec='black', linewidth=2)
        ax.scatter([0], [0], s=100, c='black', zorder=5)
        
        # Title and value
        ax.text(0, -0.3, title, ha='center', fontsize=11, fontweight='bold')
        ax.text(0, 0.6, f'{value:.2f}', ha='center', fontsize=14, fontweight='bold')
    
    def _assess_health(self, risk_report) -> dict:
        """Assess overall system health."""
        # Simple scoring
        score = (
            (1 - risk_report.debt_rank) * 0.25 +
            (1 - risk_report.systemic_risk_index) * 0.25 +
            risk_report.liquidity_index * 0.25 +
            (1 - risk_report.stress_index) * 0.25
        )
        
        if score > 0.7:
            return {
                'status': 'HEALTHY',
                'grade': 'A',
                'color': '#2ecc71',
                'message': 'System is operating within normal parameters. Continue monitoring.'
            }
        elif score > 0.5:
            return {
                'status': 'CAUTION',
                'grade': 'B',
                'color': '#f39c12',
                'message': 'Elevated risk levels detected. Consider preventive measures.'
            }
        elif score > 0.3:
            return {
                'status': 'WARNING',
                'grade': 'C',
                'color': '#e67e22',
                'message': 'High risk of contagion. Immediate intervention recommended.'
            }
        else:
            return {
                'status': 'CRITICAL',
                'grade': 'D',
                'color': '#e74c3c',
                'message': 'System instability detected. Emergency protocols advised.'
            }


def print_simulation_header(scenario: str, num_banks: int, num_steps: int):
    """Print a formatted simulation header."""
    print("\n" + "═" * 70)
    print("║" + " " * 20 + "FinSim-MAPPO Simulation" + " " * 23 + "║")
    print("║" + " " * 10 + "Network-Based Multi-Agent RL for Financial Stability" + " " * 5 + "║")
    print("═" * 70)
    print(f"║  Scenario:    {scenario:<54s}║")
    print(f"║  Banks:       {num_banks:<54d}║")
    print(f"║  Steps:       {num_steps:<54d}║")
    print("═" * 70 + "\n")


def print_step_progress(step: int, total_steps: int, defaults: int, 
                        stressed: int, price: float, avg_cr: float):
    """Print step progress in a formatted way."""
    bar_len = 20
    filled = int(bar_len * step / total_steps)
    bar = '▓' * filled + '░' * (bar_len - filled)
    
    # Status indicators
    def_indicator = '🔴' if defaults > 0 else '🟢'
    stress_indicator = '🟡' if stressed > 0 else '🟢'
    price_indicator = '📉' if price < 0.9 else ('📈' if price > 1.1 else '📊')
    
    print(f"\r  Step [{bar}] {step:3d}/{total_steps} │ "
          f"{def_indicator} Def: {defaults:2d} │ "
          f"{stress_indicator} Str: {stressed:2d} │ "
          f"{price_indicator} P: {price:.3f} │ "
          f"CR: {avg_cr*100:5.2f}%", end='', flush=True)


def print_simulation_summary(final_stats, market_state, total_rewards: dict,
                             risk_report=None):
    """Print formatted simulation summary."""
    print("\n\n" + "═" * 70)
    print("║" + " " * 22 + "SIMULATION COMPLETE" + " " * 25 + "║")
    print("═" * 70)
    
    # Network Summary
    print("\n  📊 NETWORK STATUS")
    print("  " + "─" * 40)
    print(f"  │ Active Banks:    {final_stats.num_banks - final_stats.num_defaulted - final_stats.num_stressed:3d}")
    print(f"  │ Stressed Banks:  {final_stats.num_stressed:3d}")
    print(f"  │ Defaulted Banks: {final_stats.num_defaulted:3d}")
    print(f"  │ Total Exposure:  {final_stats.total_exposure:,.0f}")
    print(f"  │ Avg Cap Ratio:   {final_stats.avg_capital_ratio*100:.2f}%")
    
    # Market Summary
    print("\n  📈 MARKET STATUS")
    print("  " + "─" * 40)
    print(f"  │ Asset Price:     {market_state.asset_price:.4f}")
    print(f"  │ Volatility:      {market_state.volatility:.4f}")
    print(f"  │ Liquidity:       {market_state.liquidity_index:.2f}")
    condition = market_state.condition.value if hasattr(market_state.condition, 'value') else str(market_state.condition)
    print(f"  │ Condition:       {condition}")
    
    # Rewards Summary
    print("\n  💰 REWARD SUMMARY")
    print("  " + "─" * 40)
    avg_reward = np.mean(list(total_rewards.values()))
    std_reward = np.std(list(total_rewards.values()))
    total = sum(total_rewards.values())
    print(f"  │ Total Reward:    {total:,.2f}")
    print(f"  │ Average:         {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"  │ Best Agent:      {max(total_rewards, key=total_rewards.get)} ({max(total_rewards.values()):.2f})")
    print(f"  │ Worst Agent:     {min(total_rewards, key=total_rewards.get)} ({min(total_rewards.values()):.2f})")
    
    # Risk Summary (if available)
    if risk_report:
        print("\n  ⚠️  RISK METRICS")
        print("  " + "─" * 40)
        print(f"  │ DebtRank:        {risk_report.debt_rank:.4f}")
        print(f"  │ Systemic Risk:   {risk_report.systemic_risk_index:.4f}")
        print(f"  │ Liquidity Idx:   {risk_report.liquidity_index:.4f}")
        print(f"  │ Stress Index:    {risk_report.stress_index:.4f}")
        print(f"  │ Cascade Pot:     {risk_report.cascade_potential*100:.1f}%")
        
        if risk_report.systemically_important_banks:
            print(f"  │ SIBs:            {risk_report.systemically_important_banks}")
    
    print("\n" + "═" * 70 + "\n")
