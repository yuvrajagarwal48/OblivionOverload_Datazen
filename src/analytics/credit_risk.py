"""
Predictive Credit Risk Layer for FinSim-MAPPO.
Independent risk estimation using supervised learning.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import json

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class RiskRating(str, Enum):
    """Credit risk ratings."""
    AAA = "AAA"
    AA = "AA"
    A = "A"
    BBB = "BBB"
    BB = "BB"
    B = "B"
    CCC = "CCC"
    D = "D"


@dataclass
class CreditRiskOutput:
    """Output from credit risk model."""
    entity_id: int
    timestamp: str
    
    # Core Risk Metrics
    probability_of_default: float = 0.0  # PD (0-1)
    loss_given_default: float = 0.45  # LGD (0-1)
    exposure_at_default: float = 0.0  # EAD
    expected_loss: float = 0.0  # EL = PD * LGD * EAD
    
    # Additional Indicators
    stress_amplification_factor: float = 1.0  # How much stress is amplified
    contagion_vulnerability: float = 0.0  # Susceptibility to contagion
    systemic_importance: float = 0.0  # Contribution to systemic risk
    
    # Credit Rating
    rating: RiskRating = RiskRating.BBB
    rating_outlook: str = "stable"  # positive, stable, negative
    
    # Confidence
    model_confidence: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['rating'] = self.rating.value
        return result
    
    @staticmethod
    def pd_to_rating(pd: float) -> RiskRating:
        """Convert PD to credit rating."""
        if pd < 0.0001:
            return RiskRating.AAA
        elif pd < 0.0004:
            return RiskRating.AA
        elif pd < 0.001:
            return RiskRating.A
        elif pd < 0.004:
            return RiskRating.BBB
        elif pd < 0.01:
            return RiskRating.BB
        elif pd < 0.04:
            return RiskRating.B
        elif pd < 0.15:
            return RiskRating.CCC
        else:
            return RiskRating.D


@dataclass
class RiskFeatures:
    """Features for credit risk prediction."""
    # Balance Sheet Features
    capital_ratio: float = 0.12
    leverage_ratio: float = 10.0
    liquidity_ratio: float = 0.1
    asset_quality: float = 0.95
    
    # Network Features
    degree_centrality: float = 0.1
    betweenness_centrality: float = 0.0
    eigenvector_centrality: float = 0.1
    clustering_coefficient: float = 0.3
    
    # Exposure Features
    concentration_ratio: float = 0.2
    largest_exposure_ratio: float = 0.1
    interbank_ratio: float = 0.3
    
    # Market/Stress Features
    market_volatility: float = 0.02
    stress_index: float = 0.0
    sector_stress: float = 0.0
    
    # Infrastructure Features
    ccp_membership_count: int = 1
    margin_coverage_ratio: float = 1.5
    exchange_congestion: float = 0.1
    
    # Historical Features
    days_stressed: int = 0
    margin_call_count: int = 0
    recent_default_exposure: float = 0.0
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        return np.array([
            self.capital_ratio,
            self.leverage_ratio / 30,  # Normalize
            self.liquidity_ratio,
            self.asset_quality,
            self.degree_centrality,
            self.betweenness_centrality,
            self.eigenvector_centrality,
            self.clustering_coefficient,
            self.concentration_ratio,
            self.largest_exposure_ratio,
            self.interbank_ratio,
            self.market_volatility * 10,  # Scale
            self.stress_index,
            self.sector_stress,
            self.ccp_membership_count / 5,  # Normalize
            min(self.margin_coverage_ratio / 2, 1),
            self.exchange_congestion,
            min(self.days_stressed / 10, 1),
            min(self.margin_call_count / 5, 1),
            min(self.recent_default_exposure / 1e6, 1)
        ], dtype=np.float32)
    
    @classmethod
    def from_bank(cls, bank: Any, network: Any = None, 
                  market: Any = None, ccp: Any = None) -> 'RiskFeatures':
        """Extract features from bank and environment."""
        bs = bank.balance_sheet
        
        # Balance sheet features
        capital_ratio = bank.capital_ratio
        leverage = bs.total_assets / max(bs.equity, 1e-8)
        liquidity = bs.cash / max(bs.total_liabilities, 1e-8)
        
        # Network features (if available)
        degree = 0.0
        betweenness = 0.0
        eigenvector = 0.0
        clustering = 0.0
        
        if network and hasattr(network, 'graph'):
            try:
                import networkx as nx
                g = network.graph
                degree = nx.degree_centrality(g).get(bank.bank_id, 0)
                betweenness = nx.betweenness_centrality(g).get(bank.bank_id, 0)
                eigenvector = nx.eigenvector_centrality_numpy(g).get(bank.bank_id, 0)
                clustering = nx.clustering(g).get(bank.bank_id, 0)
            except Exception:
                pass
        
        # Exposure features
        if hasattr(bank, 'exposures') and bank.exposures:
            total_exp = sum(bank.exposures.values())
            max_exp = max(bank.exposures.values())
            concentration = sum((e/total_exp)**2 for e in bank.exposures.values()) if total_exp > 0 else 0
            largest_ratio = max_exp / max(bs.equity, 1e-8)
        else:
            concentration = 0.0
            largest_ratio = 0.0
        
        interbank_total = sum(bs.interbank_assets.values()) if isinstance(bs.interbank_assets, dict) else bs.interbank_assets
        interbank = interbank_total / max(bs.total_assets, 1e-8)
        
        # Market features
        volatility = 0.02
        stress = 0.0
        if market:
            if hasattr(market, 'volatility'):
                volatility = market.volatility
            if hasattr(market, 'get_state'):
                state = market.get_state()
                stress = 1 - state.liquidity_index if hasattr(state, 'liquidity_index') else 0
        
        # CCP features
        margin_coverage = 1.0
        if ccp and hasattr(ccp, 'margin_accounts'):
            if bank.bank_id in ccp.margin_accounts:
                account = ccp.margin_accounts[bank.bank_id]
                margin_coverage = account.total_margin / max(account.initial_margin, 1e-8)
        
        return cls(
            capital_ratio=capital_ratio,
            leverage_ratio=leverage,
            liquidity_ratio=liquidity,
            asset_quality=0.95,  # Default assumption
            degree_centrality=degree,
            betweenness_centrality=betweenness,
            eigenvector_centrality=eigenvector,
            clustering_coefficient=clustering,
            concentration_ratio=concentration,
            largest_exposure_ratio=largest_ratio,
            interbank_ratio=interbank,
            market_volatility=volatility,
            stress_index=stress,
            margin_coverage_ratio=margin_coverage,
            days_stressed=getattr(bank, 'days_stressed', 0),
            margin_call_count=getattr(bank, 'margin_call_count', 0),
            recent_default_exposure=getattr(bank, 'exposure_to_defaults', 0)
        )


class RuleBasedRiskModel:
    """
    Simple rule-based credit risk model.
    Used as fallback when ML model unavailable.
    """
    
    def __init__(self):
        # Risk factor weights
        self.weights = {
            'capital': 0.25,
            'liquidity': 0.20,
            'leverage': 0.15,
            'network': 0.15,
            'concentration': 0.10,
            'stress': 0.15
        }
    
    def predict(self, features: RiskFeatures) -> CreditRiskOutput:
        """Predict credit risk using rules."""
        # Capital score (higher is better)
        capital_score = min(features.capital_ratio / 0.15, 1.0)
        
        # Liquidity score
        liquidity_score = min(features.liquidity_ratio / 0.15, 1.0)
        
        # Leverage score (lower is better)
        leverage_score = max(0, 1 - features.leverage_ratio / 30)
        
        # Network vulnerability (more central = more risky contagion)
        network_score = 1 - (features.betweenness_centrality * 0.5 + 
                            features.degree_centrality * 0.3 +
                            features.eigenvector_centrality * 0.2)
        
        # Concentration score (lower concentration = better)
        concentration_score = 1 - features.concentration_ratio
        
        # Stress score (lower stress = better)
        stress_score = 1 - (features.stress_index * 0.5 + 
                           features.market_volatility * 10 * 0.3 +
                           min(features.days_stressed / 10, 1) * 0.2)
        
        # Combine scores
        composite_score = (
            capital_score * self.weights['capital'] +
            liquidity_score * self.weights['liquidity'] +
            leverage_score * self.weights['leverage'] +
            network_score * self.weights['network'] +
            concentration_score * self.weights['concentration'] +
            stress_score * self.weights['stress']
        )
        
        # Convert to PD (inverse exponential relationship)
        # Score of 1.0 -> PD ~ 0.001, Score of 0.0 -> PD ~ 0.5
        pd = 0.5 * np.exp(-5 * composite_score)
        pd = np.clip(pd, 0.0001, 0.99)
        
        # LGD based on asset quality and market conditions
        lgd = 0.45 + 0.2 * (1 - features.asset_quality) + 0.1 * features.stress_index
        lgd = np.clip(lgd, 0.20, 0.80)
        
        # Stress amplification
        amplification = 1.0 + features.betweenness_centrality * 2 + features.eigenvector_centrality
        
        # Contagion vulnerability
        vulnerability = (features.interbank_ratio * 0.4 + 
                        features.degree_centrality * 0.3 +
                        (1 - features.margin_coverage_ratio / 2) * 0.3)
        
        # Systemic importance
        systemic = (features.eigenvector_centrality * 0.4 +
                   features.betweenness_centrality * 0.3 +
                   features.degree_centrality * 0.3)
        
        return CreditRiskOutput(
            entity_id=0,
            timestamp=datetime.now().isoformat(),
            probability_of_default=float(pd),
            loss_given_default=float(lgd),
            exposure_at_default=0.0,
            expected_loss=0.0,
            stress_amplification_factor=float(amplification),
            contagion_vulnerability=float(np.clip(vulnerability, 0, 1)),
            systemic_importance=float(np.clip(systemic, 0, 1)),
            rating=CreditRiskOutput.pd_to_rating(pd),
            rating_outlook='stable' if features.stress_index < 0.3 else 'negative',
            model_confidence=0.7
        )


if TORCH_AVAILABLE:
    class NeuralRiskModel(nn.Module):
        """
        Neural network-based credit risk model.
        """
        
        def __init__(self, input_dim: int = 20, hidden_dim: int = 64):
            super().__init__()
            
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
            )
            
            # Output heads
            self.pd_head = nn.Sequential(
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
            
            self.lgd_head = nn.Sequential(
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
            
            self.amplification_head = nn.Sequential(
                nn.Linear(hidden_dim // 2, 1),
                nn.Softplus()
            )
        
        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            features = self.network(x)
            pd = self.pd_head(features)
            lgd = self.lgd_head(features) * 0.6 + 0.2  # Scale to [0.2, 0.8]
            amplification = self.amplification_head(features) + 1  # Min 1.0
            return pd, lgd, amplification


class CreditRiskLayer:
    """
    Main credit risk prediction layer.
    Provides risk estimates for all entities.
    """
    
    def __init__(self, use_neural: bool = False, model_path: Optional[str] = None):
        self.use_neural = use_neural and TORCH_AVAILABLE
        
        if self.use_neural:
            self.model = NeuralRiskModel()
            if model_path:
                self.load_model(model_path)
        else:
            self.model = RuleBasedRiskModel()
        
        # Cache for efficiency
        self._risk_cache: Dict[int, CreditRiskOutput] = {}
        self._cache_timestep: int = -1
    
    def predict(self, 
                bank: Any,
                network: Any = None,
                market: Any = None,
                ccp: Any = None) -> CreditRiskOutput:
        """
        Predict credit risk for a single bank.
        """
        features = RiskFeatures.from_bank(bank, network, market, ccp)
        
        if self.use_neural:
            return self._predict_neural(features, bank.bank_id)
        else:
            output = self.model.predict(features)
            output.entity_id = bank.bank_id
            output.exposure_at_default = bank.balance_sheet.total_liabilities
            output.expected_loss = output.probability_of_default * output.loss_given_default * output.exposure_at_default
            return output
    
    def _predict_neural(self, features: RiskFeatures, entity_id: int) -> CreditRiskOutput:
        """Predict using neural model."""
        x = torch.FloatTensor(features.to_array()).unsqueeze(0)
        
        with torch.no_grad():
            pd, lgd, amp = self.model(x)
        
        pd_val = pd.item()
        lgd_val = lgd.item()
        amp_val = amp.item()
        
        return CreditRiskOutput(
            entity_id=entity_id,
            timestamp=datetime.now().isoformat(),
            probability_of_default=pd_val,
            loss_given_default=lgd_val,
            stress_amplification_factor=amp_val,
            rating=CreditRiskOutput.pd_to_rating(pd_val),
            model_confidence=0.85
        )
    
    def predict_all(self,
                    banks: Dict[int, Any],
                    network: Any = None,
                    market: Any = None,
                    ccps: Optional[List[Any]] = None,
                    timestep: int = -1) -> Dict[int, CreditRiskOutput]:
        """
        Predict credit risk for all banks.
        Uses caching for efficiency.
        """
        if timestep == self._cache_timestep and self._risk_cache:
            return self._risk_cache
        
        results = {}
        
        for bank_id, bank in banks.items():
            ccp = ccps[0] if ccps else None  # Use first CCP for now
            results[bank_id] = self.predict(bank, network, market, ccp)
        
        self._risk_cache = results
        self._cache_timestep = timestep
        
        return results
    
    def get_system_risk_summary(self, 
                                 risk_outputs: Dict[int, CreditRiskOutput]) -> Dict[str, Any]:
        """
        Aggregate individual risk outputs into system summary.
        """
        if not risk_outputs:
            return {}
        
        outputs = list(risk_outputs.values())
        
        return {
            'average_pd': np.mean([o.probability_of_default for o in outputs]),
            'max_pd': max(o.probability_of_default for o in outputs),
            'high_risk_count': sum(1 for o in outputs if o.probability_of_default > 0.05),
            'total_expected_loss': sum(o.expected_loss for o in outputs),
            'average_amplification': np.mean([o.stress_amplification_factor for o in outputs]),
            'systemic_concentration': sum(o.systemic_importance ** 2 for o in outputs),
            'rating_distribution': self._compute_rating_distribution(outputs)
        }
    
    def _compute_rating_distribution(self, 
                                     outputs: List[CreditRiskOutput]) -> Dict[str, int]:
        """Count entities in each rating bucket."""
        distribution = {r.value: 0 for r in RiskRating}
        for output in outputs:
            distribution[output.rating.value] += 1
        return distribution
    
    def integrate_with_observations(self,
                                    observations: Dict[int, np.ndarray],
                                    risk_outputs: Dict[int, CreditRiskOutput]) -> Dict[int, np.ndarray]:
        """
        Integrate risk predictions into agent observations.
        Extends observation vector with risk features.
        """
        extended_obs = {}
        
        for agent_id, obs in observations.items():
            if agent_id in risk_outputs:
                risk = risk_outputs[agent_id]
                risk_features = np.array([
                    risk.probability_of_default,
                    risk.loss_given_default,
                    risk.stress_amplification_factor - 1,  # Normalize
                    risk.contagion_vulnerability,
                    risk.systemic_importance
                ], dtype=np.float32)
                
                extended_obs[agent_id] = np.concatenate([obs, risk_features])
            else:
                # Pad with zeros if no risk output
                extended_obs[agent_id] = np.concatenate([obs, np.zeros(5)])
        
        return extended_obs
    
    def compute_reward_adjustment(self,
                                  base_reward: float,
                                  risk_output: CreditRiskOutput,
                                  previous_risk: Optional[CreditRiskOutput] = None) -> float:
        """
        Adjust reward based on risk changes.
        Encourages risk-aware behavior.
        """
        # Penalty for high PD
        pd_penalty = -risk_output.probability_of_default * 10
        
        # Penalty for systemic risk contribution
        systemic_penalty = -risk_output.systemic_importance * 2
        
        # Bonus for risk improvement
        improvement_bonus = 0
        if previous_risk:
            pd_improvement = previous_risk.probability_of_default - risk_output.probability_of_default
            improvement_bonus = pd_improvement * 20  # Reward for reducing risk
        
        adjusted_reward = base_reward + pd_penalty + systemic_penalty + improvement_bonus
        
        return adjusted_reward
    
    def save_model(self, path: str) -> None:
        """Save neural model weights."""
        if self.use_neural and hasattr(self.model, 'state_dict'):
            torch.save(self.model.state_dict(), path)
    
    def load_model(self, path: str) -> None:
        """Load neural model weights."""
        if self.use_neural and hasattr(self.model, 'load_state_dict'):
            self.model.load_state_dict(torch.load(path))
            self.model.eval()
