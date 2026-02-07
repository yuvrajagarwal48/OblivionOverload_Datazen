"""
Comprehensive Logging and Monitoring Module for FinSim-MAPPO.
Tracks training diagnostics, environment metrics, and system health.
"""

import numpy as np
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import deque
import threading
import queue


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""
    episode: int
    timestamp: str
    
    # Rewards
    total_reward: float = 0.0
    mean_reward: float = 0.0
    std_reward: float = 0.0
    min_reward: float = 0.0
    max_reward: float = 0.0
    
    # Episode info
    episode_length: int = 0
    curriculum_stage: int = 0
    
    # System metrics
    default_count: int = 0
    default_rate: float = 0.0
    stressed_count: int = 0
    final_avg_capital_ratio: float = 0.0
    
    # Market metrics
    final_price: float = 1.0
    min_price: float = 1.0
    max_volatility: float = 0.0
    
    # Infrastructure metrics
    avg_exchange_congestion: float = 0.0
    ccp_stress_level: float = 0.0
    total_margin_calls: float = 0.0
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainingMetrics:
    """Metrics from a training update."""
    update_step: int
    timestamp: str
    
    # Loss values
    actor_loss: float = 0.0
    critic_loss: float = 0.0
    total_loss: float = 0.0
    
    # Policy metrics
    policy_entropy: float = 0.0
    kl_divergence: float = 0.0
    clip_fraction: float = 0.0
    
    # Value estimates
    explained_variance: float = 0.0
    value_loss: float = 0.0
    advantage_mean: float = 0.0
    advantage_std: float = 0.0
    
    # Learning rate
    learning_rate: float = 0.0
    entropy_coef: float = 0.0
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StepMetrics:
    """Metrics for a single environment step."""
    episode: int
    step: int
    
    # Market state
    asset_price: float = 1.0
    volatility: float = 0.02
    liquidity_index: float = 1.0
    
    # Network state
    num_defaults: int = 0
    num_stressed: int = 0
    avg_capital_ratio: float = 0.0
    total_exposure: float = 0.0
    
    # Infrastructure
    exchange_congestion: float = 0.0
    ccp_margin_calls: float = 0.0
    settlement_delays: float = 0.0
    
    # Actions
    avg_lend_ratio: float = 0.0
    avg_hoard_ratio: float = 0.0
    avg_sell_ratio: float = 0.0
    
    # Rewards
    total_reward: float = 0.0
    
    def to_dict(self) -> dict:
        return asdict(self)


class MetricsBuffer:
    """Thread-safe buffer for metrics."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._buffer: deque = deque(maxlen=max_size)
        self._lock = threading.Lock()
    
    def add(self, metric: Any) -> None:
        with self._lock:
            self._buffer.append(metric)
    
    def get_all(self) -> List[Any]:
        with self._lock:
            return list(self._buffer)
    
    def get_recent(self, n: int) -> List[Any]:
        with self._lock:
            return list(self._buffer)[-n:]
    
    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()
    
    def __len__(self) -> int:
        return len(self._buffer)


class TrainingLogger:
    """
    Comprehensive logger for training diagnostics.
    
    Records:
    - Episode rewards and lengths
    - Default rates and system stability
    - Loss values and policy entropy
    - KL divergence and clip fractions
    """
    
    def __init__(self, log_dir: str = "logs", experiment_name: Optional[str] = None):
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(log_dir) / self.experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Metric buffers
        self.episode_metrics = MetricsBuffer()
        self.training_metrics = MetricsBuffer()
        self.step_metrics = MetricsBuffer(max_size=50000)
        
        # Current episode tracking
        self.current_episode = 0
        self.current_step = 0
        
        # File handles
        self._episode_file = None
        self._training_file = None
        self._step_file = None
        
        # Initialize files
        self._init_files()
        
        # Aggregate statistics
        self.total_episodes = 0
        self.total_steps = 0
        self.best_mean_reward = float('-inf')
        self.best_default_rate = 1.0
    
    def _init_files(self) -> None:
        """Initialize CSV log files."""
        # Episode metrics file
        episode_path = self.log_dir / "episode_metrics.csv"
        self._episode_file = open(episode_path, 'w', newline='')
        self._episode_writer = csv.DictWriter(
            self._episode_file,
            fieldnames=list(EpisodeMetrics.__dataclass_fields__.keys())
        )
        self._episode_writer.writeheader()
        
        # Training metrics file
        training_path = self.log_dir / "training_metrics.csv"
        self._training_file = open(training_path, 'w', newline='')
        self._training_writer = csv.DictWriter(
            self._training_file,
            fieldnames=list(TrainingMetrics.__dataclass_fields__.keys())
        )
        self._training_writer.writeheader()
    
    def log_episode(self, metrics: EpisodeMetrics) -> None:
        """Log episode metrics."""
        self.episode_metrics.add(metrics)
        self._episode_writer.writerow(metrics.to_dict())
        self._episode_file.flush()
        
        self.total_episodes += 1
        self.current_episode = metrics.episode
        
        # Update best metrics
        if metrics.mean_reward > self.best_mean_reward:
            self.best_mean_reward = metrics.mean_reward
        if metrics.default_rate < self.best_default_rate:
            self.best_default_rate = metrics.default_rate
    
    def log_training_update(self, metrics: TrainingMetrics) -> None:
        """Log training update metrics."""
        self.training_metrics.add(metrics)
        self._training_writer.writerow(metrics.to_dict())
        self._training_file.flush()
    
    def log_step(self, metrics: StepMetrics) -> None:
        """Log step metrics (in memory only for efficiency)."""
        self.step_metrics.add(metrics)
        self.total_steps += 1
        self.current_step = metrics.step
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of logged metrics."""
        recent_episodes = self.episode_metrics.get_recent(100)
        
        if not recent_episodes:
            return {}
        
        return {
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,
            'best_mean_reward': self.best_mean_reward,
            'best_default_rate': self.best_default_rate,
            'recent_mean_reward': np.mean([e.mean_reward for e in recent_episodes]),
            'recent_default_rate': np.mean([e.default_rate for e in recent_episodes]),
            'recent_episode_length': np.mean([e.episode_length for e in recent_episodes])
        }
    
    def save_summary(self) -> None:
        """Save summary to JSON file."""
        summary = self.get_summary()
        summary['experiment_name'] = self.experiment_name
        summary['timestamp'] = datetime.now().isoformat()
        
        summary_path = self.log_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def close(self) -> None:
        """Close file handles."""
        self.save_summary()
        if self._episode_file:
            self._episode_file.close()
        if self._training_file:
            self._training_file.close()


class EnvironmentLogger:
    """
    Logger for environment diagnostics.
    
    Records:
    - Price trajectories
    - Margin calls
    - Congestion levels
    - Clearing failures
    """
    
    def __init__(self, log_dir: str = "logs/environment"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Time series data
        self.price_history: List[float] = []
        self.volatility_history: List[float] = []
        self.default_history: List[int] = []
        self.stressed_history: List[int] = []
        self.capital_ratio_history: List[float] = []
        
        # Infrastructure metrics
        self.congestion_history: List[float] = []
        self.margin_call_history: List[float] = []
        self.settlement_delay_history: List[float] = []
        
        # Events
        self.clearing_failures: List[Dict] = []
        self.margin_call_events: List[Dict] = []
        self.default_events: List[Dict] = []
        
        self.current_step = 0
    
    def log_market_state(self, price: float, volatility: float, 
                         liquidity: float, step: int) -> None:
        """Log market state."""
        self.price_history.append(price)
        self.volatility_history.append(volatility)
        self.current_step = step
    
    def log_network_state(self, defaults: int, stressed: int, 
                          avg_capital_ratio: float, step: int) -> None:
        """Log network state."""
        self.default_history.append(defaults)
        self.stressed_history.append(stressed)
        self.capital_ratio_history.append(avg_capital_ratio)
    
    def log_infrastructure_state(self, congestion: float, margin_calls: float,
                                  settlement_delay: float, step: int) -> None:
        """Log infrastructure state."""
        self.congestion_history.append(congestion)
        self.margin_call_history.append(margin_calls)
        self.settlement_delay_history.append(settlement_delay)
    
    def log_clearing_failure(self, step: int, member_id: int, 
                              loss_amount: float, details: Dict) -> None:
        """Log a clearing failure event."""
        self.clearing_failures.append({
            'step': step,
            'member_id': member_id,
            'loss_amount': loss_amount,
            'details': details,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_margin_call(self, step: int, member_id: int, 
                         amount: float, met: bool) -> None:
        """Log a margin call event."""
        self.margin_call_events.append({
            'step': step,
            'member_id': member_id,
            'amount': amount,
            'met': met,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_default_event(self, step: int, bank_id: int, 
                           equity: float, reason: str) -> None:
        """Log a bank default event."""
        self.default_events.append({
            'step': step,
            'bank_id': bank_id,
            'equity_at_default': equity,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_time_series_data(self) -> Dict[str, List]:
        """Get all time series data for visualization."""
        return {
            'steps': list(range(len(self.price_history))),
            'prices': self.price_history,
            'volatilities': self.volatility_history,
            'defaults': self.default_history,
            'stressed': self.stressed_history,
            'capital_ratios': self.capital_ratio_history,
            'congestion': self.congestion_history,
            'margin_calls': self.margin_call_history,
            'settlement_delays': self.settlement_delay_history
        }
    
    def save(self, filename: str = "environment_log.json") -> None:
        """Save all logged data to file."""
        data = {
            'time_series': self.get_time_series_data(),
            'events': {
                'clearing_failures': self.clearing_failures,
                'margin_calls': self.margin_call_events,
                'defaults': self.default_events
            },
            'metadata': {
                'total_steps': self.current_step,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        filepath = self.log_dir / filename
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def reset(self) -> None:
        """Reset all logged data."""
        self.price_history.clear()
        self.volatility_history.clear()
        self.default_history.clear()
        self.stressed_history.clear()
        self.capital_ratio_history.clear()
        self.congestion_history.clear()
        self.margin_call_history.clear()
        self.settlement_delay_history.clear()
        self.clearing_failures.clear()
        self.margin_call_events.clear()
        self.default_events.clear()
        self.current_step = 0


class SystemMonitor:
    """
    Real-time system health monitoring.
    
    Detects anomalies and triggers alerts.
    """
    
    def __init__(self, alert_callback: Optional[callable] = None):
        self.alert_callback = alert_callback or self._default_alert
        
        # Thresholds
        self.price_floor = 0.1
        self.max_default_rate = 0.5
        self.max_volatility = 0.5
        self.min_capital_ratio = 0.04
        self.max_congestion = 0.9
        
        # State
        self.alerts: List[Dict] = []
        self.is_healthy = True
    
    def _default_alert(self, alert: Dict) -> None:
        """Default alert handler."""
        print(f"⚠️  ALERT: {alert['type']} - {alert['message']}")
    
    def check_market_health(self, price: float, volatility: float) -> bool:
        """Check market health indicators."""
        healthy = True
        
        if price < self.price_floor:
            alert = {
                'type': 'PRICE_FLOOR',
                'message': f'Price {price:.4f} below floor {self.price_floor}',
                'severity': 'critical',
                'timestamp': datetime.now().isoformat()
            }
            self.alerts.append(alert)
            self.alert_callback(alert)
            healthy = False
        
        if volatility > self.max_volatility:
            alert = {
                'type': 'HIGH_VOLATILITY',
                'message': f'Volatility {volatility:.4f} exceeds maximum {self.max_volatility}',
                'severity': 'warning',
                'timestamp': datetime.now().isoformat()
            }
            self.alerts.append(alert)
            self.alert_callback(alert)
            healthy = False
        
        return healthy
    
    def check_network_health(self, default_rate: float, 
                              avg_capital_ratio: float) -> bool:
        """Check network health indicators."""
        healthy = True
        
        if default_rate > self.max_default_rate:
            alert = {
                'type': 'HIGH_DEFAULT_RATE',
                'message': f'Default rate {default_rate:.2%} exceeds maximum {self.max_default_rate:.2%}',
                'severity': 'critical',
                'timestamp': datetime.now().isoformat()
            }
            self.alerts.append(alert)
            self.alert_callback(alert)
            healthy = False
        
        if avg_capital_ratio < self.min_capital_ratio:
            alert = {
                'type': 'LOW_CAPITAL',
                'message': f'Avg capital ratio {avg_capital_ratio:.2%} below minimum {self.min_capital_ratio:.2%}',
                'severity': 'warning',
                'timestamp': datetime.now().isoformat()
            }
            self.alerts.append(alert)
            self.alert_callback(alert)
            healthy = False
        
        return healthy
    
    def check_infrastructure_health(self, congestion: float, 
                                     ccp_status: str) -> bool:
        """Check infrastructure health."""
        healthy = True
        
        if congestion > self.max_congestion:
            alert = {
                'type': 'HIGH_CONGESTION',
                'message': f'Exchange congestion {congestion:.2%} exceeds maximum',
                'severity': 'warning',
                'timestamp': datetime.now().isoformat()
            }
            self.alerts.append(alert)
            self.alert_callback(alert)
            healthy = False
        
        if ccp_status == 'failed':
            alert = {
                'type': 'CCP_FAILURE',
                'message': 'Central counterparty has failed',
                'severity': 'critical',
                'timestamp': datetime.now().isoformat()
            }
            self.alerts.append(alert)
            self.alert_callback(alert)
            healthy = False
        
        return healthy
    
    def check_numerical_stability(self, values: Dict[str, float]) -> bool:
        """Check for numerical stability issues."""
        healthy = True
        
        for name, value in values.items():
            if np.isnan(value) or np.isinf(value):
                alert = {
                    'type': 'NUMERICAL_INSTABILITY',
                    'message': f'{name} has invalid value: {value}',
                    'severity': 'critical',
                    'timestamp': datetime.now().isoformat()
                }
                self.alerts.append(alert)
                self.alert_callback(alert)
                healthy = False
        
        return healthy
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status."""
        return {
            'is_healthy': self.is_healthy,
            'total_alerts': len(self.alerts),
            'critical_alerts': sum(1 for a in self.alerts if a['severity'] == 'critical'),
            'warning_alerts': sum(1 for a in self.alerts if a['severity'] == 'warning'),
            'recent_alerts': self.alerts[-10:]
        }
    
    def clear_alerts(self) -> None:
        """Clear all alerts."""
        self.alerts.clear()
        self.is_healthy = True
