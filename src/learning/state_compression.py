"""
Global State Compression for MAPPO Critic.
Provides scalable state representation for centralized training.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass


@dataclass
class GlobalStateConfig:
    """Configuration for global state compression."""
    # Compression dimensions
    compressed_dim: int = 64
    use_attention: bool = True
    num_heads: int = 4
    
    # Feature groupings
    include_network_topology: bool = True
    include_aggregate_stats: bool = True
    include_infrastructure_state: bool = True
    include_risk_indicators: bool = True
    
    # Normalization
    normalize_features: bool = True
    clip_features: float = 10.0


class NetworkTopologyEncoder(nn.Module):
    """
    Encodes network topology into fixed-size representation.
    Uses graph-level aggregation.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 32):
        super().__init__()
        
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Aggregation layers
        self.aggregate_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # mean, max, std
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        Encode node features into graph-level representation.
        
        Args:
            node_features: [batch, num_nodes, feature_dim]
        
        Returns:
            graph_encoding: [batch, output_dim]
        """
        # Encode each node
        encoded = self.node_encoder(node_features)  # [batch, num_nodes, hidden]
        
        # Aggregate with multiple strategies
        mean_agg = encoded.mean(dim=1)
        max_agg = encoded.max(dim=1)[0]
        std_agg = encoded.std(dim=1)
        
        # Concatenate and transform
        aggregated = torch.cat([mean_agg, max_agg, std_agg], dim=-1)
        return self.aggregate_mlp(aggregated)


class AttentionAggregator(nn.Module):
    """
    Uses attention mechanism to aggregate agent observations.
    """
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int = 64, 
                 output_dim: int = 32,
                 num_heads: int = 4):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Global query for aggregation
        self.global_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        self.output_projection = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, agent_observations: torch.Tensor) -> torch.Tensor:
        """
        Aggregate agent observations using attention.
        
        Args:
            agent_observations: [batch, num_agents, obs_dim]
        
        Returns:
            aggregated: [batch, output_dim]
        """
        batch_size = agent_observations.shape[0]
        
        # Project observations
        projected = self.input_projection(agent_observations)
        
        # Expand global query for batch
        query = self.global_query.expand(batch_size, -1, -1)
        
        # Attention aggregation
        attended, _ = self.attention(query, projected, projected)
        
        # Project to output
        return self.output_projection(attended.squeeze(1))


class GlobalStateCompressor(nn.Module):
    """
    Compresses global state information for centralized critic.
    
    Combines:
    1. Aggregated agent observations
    2. Network topology features
    3. System-level risk indicators
    4. Infrastructure state
    """
    
    def __init__(self, 
                 num_agents: int,
                 obs_dim: int,
                 config: GlobalStateConfig = None):
        super().__init__()
        
        self.config = config or GlobalStateConfig()
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        
        # Calculate component dimensions
        self.topology_dim = 32 if self.config.include_network_topology else 0
        self.stats_dim = 16 if self.config.include_aggregate_stats else 0
        self.infra_dim = 16 if self.config.include_infrastructure_state else 0
        self.risk_dim = 16 if self.config.include_risk_indicators else 0
        
        # Attention-based observation aggregation
        if self.config.use_attention:
            self.obs_aggregator = AttentionAggregator(
                input_dim=obs_dim,
                hidden_dim=64,
                output_dim=32,
                num_heads=self.config.num_heads
            )
            self.obs_agg_dim = 32
        else:
            # Simple MLP aggregation
            self.obs_aggregator = nn.Sequential(
                nn.Linear(obs_dim * num_agents, 128),
                nn.ReLU(),
                nn.Linear(128, 32)
            )
            self.obs_agg_dim = 32
        
        # Topology encoder
        if self.config.include_network_topology:
            self.topology_encoder = NetworkTopologyEncoder(
                input_dim=obs_dim,
                hidden_dim=64,
                output_dim=self.topology_dim
            )
        
        # Statistics encoder
        if self.config.include_aggregate_stats:
            self.stats_encoder = nn.Sequential(
                nn.Linear(10, 32),  # 10 aggregate statistics
                nn.ReLU(),
                nn.Linear(32, self.stats_dim)
            )
        
        # Infrastructure encoder
        if self.config.include_infrastructure_state:
            self.infra_encoder = nn.Sequential(
                nn.Linear(8, 32),  # 8-dim infrastructure observation
                nn.ReLU(),
                nn.Linear(32, self.infra_dim)
            )
        
        # Risk indicators encoder
        if self.config.include_risk_indicators:
            self.risk_encoder = nn.Sequential(
                nn.Linear(6, 32),  # 6 risk indicators
                nn.ReLU(),
                nn.Linear(32, self.risk_dim)
            )
        
        # Final compression
        total_features = (self.obs_agg_dim + self.topology_dim + 
                         self.stats_dim + self.infra_dim + self.risk_dim)
        
        self.final_compression = nn.Sequential(
            nn.Linear(total_features, 128),
            nn.ReLU(),
            nn.Linear(128, self.config.compressed_dim)
        )
        
        # Normalization layers
        if self.config.normalize_features:
            self.input_norm = nn.LayerNorm(obs_dim)
            self.output_norm = nn.LayerNorm(self.config.compressed_dim)
    
    def forward(self,
                agent_observations: torch.Tensor,
                network_stats: Optional[torch.Tensor] = None,
                infrastructure_state: Optional[torch.Tensor] = None,
                risk_indicators: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compress global state.
        
        Args:
            agent_observations: [batch, num_agents, obs_dim] all agent observations
            network_stats: [batch, 10] aggregate network statistics
            infrastructure_state: [batch, 8] infrastructure state
            risk_indicators: [batch, 6] risk indicators
        
        Returns:
            compressed_state: [batch, compressed_dim]
        """
        features = []
        
        # Normalize inputs
        if self.config.normalize_features:
            agent_observations = self.input_norm(agent_observations)
            if self.config.clip_features > 0:
                agent_observations = torch.clamp(
                    agent_observations, 
                    -self.config.clip_features, 
                    self.config.clip_features
                )
        
        # Aggregate observations
        if self.config.use_attention:
            obs_agg = self.obs_aggregator(agent_observations)
        else:
            batch_size = agent_observations.shape[0]
            flat_obs = agent_observations.view(batch_size, -1)
            obs_agg = self.obs_aggregator(flat_obs)
        features.append(obs_agg)
        
        # Encode topology
        if self.config.include_network_topology:
            topology_features = self.topology_encoder(agent_observations)
            features.append(topology_features)
        
        # Encode statistics
        if self.config.include_aggregate_stats and network_stats is not None:
            stats_features = self.stats_encoder(network_stats)
            features.append(stats_features)
        elif self.config.include_aggregate_stats:
            # Compute basic stats from observations
            obs_mean = agent_observations.mean(dim=1)
            obs_std = agent_observations.std(dim=1)
            obs_max = agent_observations.max(dim=1)[0]
            basic_stats = torch.cat([
                obs_mean.mean(dim=-1, keepdim=True),
                obs_std.mean(dim=-1, keepdim=True),
                obs_max.mean(dim=-1, keepdim=True)
            ], dim=-1)
            # Pad to 10 dims
            padding = torch.zeros(basic_stats.shape[0], 7, device=basic_stats.device)
            basic_stats = torch.cat([basic_stats, padding], dim=-1)
            features.append(self.stats_encoder(basic_stats))
        
        # Encode infrastructure
        if self.config.include_infrastructure_state and infrastructure_state is not None:
            infra_features = self.infra_encoder(infrastructure_state)
            features.append(infra_features)
        elif self.config.include_infrastructure_state:
            # Use zeros as placeholder
            batch_size = agent_observations.shape[0]
            infra_placeholder = torch.zeros(batch_size, 8, device=agent_observations.device)
            features.append(self.infra_encoder(infra_placeholder))
        
        # Encode risk indicators
        if self.config.include_risk_indicators and risk_indicators is not None:
            risk_features = self.risk_encoder(risk_indicators)
            features.append(risk_features)
        elif self.config.include_risk_indicators:
            # Compute risk from observations
            batch_size = agent_observations.shape[0]
            risk_placeholder = torch.zeros(batch_size, 6, device=agent_observations.device)
            features.append(self.risk_encoder(risk_placeholder))
        
        # Concatenate and compress
        combined = torch.cat(features, dim=-1)
        compressed = self.final_compression(combined)
        
        if self.config.normalize_features:
            compressed = self.output_norm(compressed)
        
        return compressed
    
    @property
    def output_dim(self) -> int:
        """Get output dimension of compressed state."""
        return self.config.compressed_dim


def compute_network_statistics(network: Any) -> np.ndarray:
    """
    Compute aggregate network statistics for global state.
    
    Returns 10-dimensional vector:
    - num_defaulted / num_banks
    - num_stressed / num_banks
    - total_exposure / initial_exposure
    - avg_capital_ratio
    - std_capital_ratio
    - max_interconnectedness
    - avg_interconnectedness
    - concentration_index (HHI)
    - liquidity_ratio
    - solvency_ratio
    """
    stats = network.get_network_stats()
    num_banks = len(network.banks)
    
    # Capital ratios
    capital_ratios = []
    liquidity_ratios = []
    exposures = []
    
    for bank in network.banks.values():
        bs = bank.balance_sheet
        capital_ratio = bs.equity / max(bs.total_assets, 1)
        capital_ratios.append(capital_ratio)
        
        liquidity = bs.cash / max(bs.total_liabilities, 1)
        liquidity_ratios.append(liquidity)
        
        exposures.append(sum(network.liability_matrix[bank.bank_id]))
    
    total_exposure = sum(exposures)
    
    # Concentration (HHI)
    if total_exposure > 0:
        shares = [e / total_exposure for e in exposures]
        hhi = sum(s ** 2 for s in shares)
    else:
        hhi = 0
    
    # Interconnectedness
    degrees = [network.graph.degree(i) for i in range(num_banks)]
    
    return np.array([
        stats.num_defaulted / max(num_banks, 1),
        stats.num_stressed / max(num_banks, 1),
        total_exposure / max(stats.total_exposure, 1),
        np.mean(capital_ratios),
        np.std(capital_ratios),
        max(degrees) / max(num_banks - 1, 1),
        np.mean(degrees) / max(num_banks - 1, 1),
        hhi,
        np.mean(liquidity_ratios),
        np.mean([1 for cr in capital_ratios if cr > 0.08]) / max(num_banks, 1)  # Solvency
    ], dtype=np.float32)


def compute_risk_indicators(network: Any, market: Any) -> np.ndarray:
    """
    Compute system-wide risk indicators.
    
    Returns 6-dimensional vector:
    - systemic_risk_score
    - contagion_potential
    - liquidity_stress
    - market_volatility
    - default_cascade_risk
    - recovery_capacity
    """
    stats = network.get_network_stats()
    num_banks = len(network.banks)
    
    # Systemic risk: combination of defaults and stress
    systemic = (stats.num_defaulted + 0.5 * stats.num_stressed) / max(num_banks, 1)
    
    # Contagion potential: interconnectedness * stressed ratio
    avg_degree = np.mean([network.graph.degree(i) for i in range(num_banks)])
    contagion = avg_degree / max(num_banks - 1, 1) * (stats.num_stressed / max(num_banks, 1))
    
    # Liquidity stress
    low_liquidity = sum(
        1 for bank in network.banks.values()
        if bank.balance_sheet.cash / max(bank.balance_sheet.total_liabilities, 1) < 0.1
    )
    liquidity_stress = low_liquidity / max(num_banks, 1)
    
    # Market volatility (if available)
    if market and hasattr(market, 'volatility'):
        volatility = market.volatility
    else:
        volatility = 0.2  # Default
    
    # Default cascade risk
    if stats.num_defaulted > 0:
        cascade_risk = min(1.0, stats.num_defaulted / max(num_banks, 1) * 2)
    else:
        cascade_risk = contagion * 0.5
    
    # Recovery capacity: healthy banks' capital
    healthy_capital = sum(
        bank.balance_sheet.equity
        for bank in network.banks.values()
        if bank.status.value == 'healthy'
    )
    total_capital = sum(bank.balance_sheet.equity for bank in network.banks.values())
    recovery = healthy_capital / max(total_capital, 1)
    
    return np.array([
        systemic,
        contagion,
        liquidity_stress,
        volatility,
        cascade_risk,
        recovery
    ], dtype=np.float32)


class CentralizedCriticWithCompression(nn.Module):
    """
    Centralized critic that uses global state compression.
    """
    
    def __init__(self,
                 num_agents: int,
                 obs_dim: int,
                 hidden_dim: int = 256,
                 global_config: GlobalStateConfig = None):
        super().__init__()
        
        self.compressor = GlobalStateCompressor(
            num_agents=num_agents,
            obs_dim=obs_dim,
            config=global_config
        )
        
        compressed_dim = self.compressor.output_dim
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(compressed_dim + obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self,
                local_obs: torch.Tensor,
                all_observations: torch.Tensor,
                network_stats: Optional[torch.Tensor] = None,
                infrastructure_state: Optional[torch.Tensor] = None,
                risk_indicators: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute value estimate.
        
        Args:
            local_obs: [batch, obs_dim] local agent observation
            all_observations: [batch, num_agents, obs_dim] all observations
            network_stats: optional aggregate stats
            infrastructure_state: optional infrastructure state
            risk_indicators: optional risk indicators
        
        Returns:
            value: [batch, 1]
        """
        # Compress global state
        compressed = self.compressor(
            all_observations,
            network_stats,
            infrastructure_state,
            risk_indicators
        )
        
        # Combine with local observation
        combined = torch.cat([compressed, local_obs], dim=-1)
        
        return self.value_head(combined)
