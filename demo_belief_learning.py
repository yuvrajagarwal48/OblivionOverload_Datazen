"""
Integration Demo: Private Memory & Belief Learning System.

This demo shows how the belief learning components work together
to create subjective agents with private memory, beliefs, and trust.
"""

import numpy as np
from typing import Dict, List, Optional

# Import all belief learning components
from src.agents.memory import (
    InteractionType, InteractionRecord, PrivateMemory, MemoryManager
)
from src.agents.beliefs import (
    BeliefConfidence, CounterpartyBelief, MarketPrior, BeliefSystem, BeliefManager
)
from src.agents.belief_integration import (
    BeliefObservationBuilder, BeliefFilter, AdaptiveTermsEngine, BeliefIntegratedObserver
)
from src.agents.belief_counterfactual import (
    BeliefCounterfactualEngine
)
from src.agents.reputation import (
    TrustDynamicsEngine, ReputationAggregator
)
from src.agents.belief_explanation import (
    BeliefLogger, BeliefExplainer, BeliefUIExporter
)
from src.agents.belief_evaluation import (
    BeliefEvaluationEngine, SafetyBounds
)


class BeliefLearningSystem:
    """
    Integrated belief learning system for FinSim-MAPPO.
    
    This system manages:
    - Private memory for each bank
    - Belief formation and updates
    - Trust network dynamics
    - Decision filtering based on beliefs
    - Counterfactual analysis
    - Calibration and safety monitoring
    - UI data export
    """
    
    def __init__(self, 
                 bank_ids: List[int],
                 belief_dim: int = 8,
                 market_prior: Optional[MarketPrior] = None):
        """
        Initialize the belief learning system.
        
        Args:
            bank_ids: List of bank identifiers
            belief_dim: Dimension of belief observation vectors
            market_prior: Default prior beliefs for new counterparties
        """
        self.bank_ids = bank_ids
        self.belief_dim = belief_dim
        self.market_prior = market_prior or MarketPrior()
        
        # Core components
        self.memory_manager = MemoryManager()  # No bank_ids needed, creates on demand
        self.belief_manager = BeliefManager(self.memory_manager)
        
        # Initialize belief systems for all banks
        self.belief_systems: Dict[int, BeliefSystem] = {}
        for bank_id in bank_ids:
            self.belief_systems[bank_id] = self.belief_manager.get_or_create_belief_system(bank_id)
        
        # Trust dynamics - pass belief systems dict
        self.trust_engine = TrustDynamicsEngine(self.belief_systems)
        
        # Per-bank components (created on demand)
        self.obs_builders: Dict[int, BeliefObservationBuilder] = {}
        self.belief_filters: Dict[int, BeliefFilter] = {}
        self.terms_engines: Dict[int, AdaptiveTermsEngine] = {}
        
        # Initialize per-bank components
        for bank_id in bank_ids:
            belief_sys = self.belief_systems[bank_id]
            self.obs_builders[bank_id] = BeliefObservationBuilder(
                belief_system=belief_sys,
                base_obs_dim=20,
                belief_dim=belief_dim
            )
            self.belief_filters[bank_id] = BeliefFilter(
                belief_system=belief_sys,
                min_trust_threshold=0.2,
                max_pd_threshold=0.3
            )
            self.terms_engines[bank_id] = AdaptiveTermsEngine(
                belief_system=belief_sys
            )
        
        # Counterfactual engines (per bank)
        self.counterfactual_engines: Dict[int, BeliefCounterfactualEngine] = {}
        for bank_id in bank_ids:
            belief_sys = self.belief_systems[bank_id]
            memory = self.memory_manager.get_or_create_memory(bank_id)
            self.counterfactual_engines[bank_id] = BeliefCounterfactualEngine(
                belief_system=belief_sys,
                private_memory=memory
            )
        
        # Evaluation and safety
        self.evaluation_engine = BeliefEvaluationEngine(
            safety_bounds=SafetyBounds()
        )
        
        # Logging for explanation
        self.belief_logger = BeliefLogger()
        self.belief_explainer = BeliefExplainer(self.belief_logger)
        
        # Current timestep
        self.timestep = 0
    
    def get_belief_system(self, bank_id: int) -> BeliefSystem:
        """Get belief system for a bank."""
        return self.belief_systems[bank_id]
    
    def record_interaction(self, 
                           bank_id: int,
                           counterparty_id: int,
                           interaction_type: InteractionType,
                           amount: float,
                           rate: Optional[float] = None,
                           collateral: Optional[float] = None,
                           metadata: Optional[Dict] = None) -> str:
        """
        Record an interaction in private memory.
        Returns interaction_id for tracking outcomes.
        """
        # Store in private memory
        memory = self.memory_manager.get_or_create_memory(bank_id)
        interaction_id = memory.record_interaction(
            counterparty_id=counterparty_id,
            interaction_type=interaction_type,
            amount=amount,
            interest_rate=rate or 0.0,
            collateral_amount=collateral or 0.0
        )
        return interaction_id
    
    def record_outcome(self,
                       bank_id: int,
                       counterparty_id: int,
                       interaction_type: InteractionType,
                       successful: bool,
                       actual_repayment: Optional[float] = None,
                       default_occurred: bool = False) -> None:
        """
        Record the outcome of an interaction and update beliefs.
        """
        # Update beliefs
        belief_system = self.belief_systems[bank_id]
        belief_system.update_from_interaction(
            counterparty_id=counterparty_id,
            default_occurred=default_occurred,
            delay_occurred=not successful and not default_occurred,
            profit_loss=actual_repayment if actual_repayment else 0.0,
            amount=actual_repayment if actual_repayment else 0.0
        )
        
        # Log belief state
        belief = belief_system.get_belief(counterparty_id)
        event = f"{'Success' if successful else 'Failure'}: {interaction_type.name}"
        self.belief_logger.log_belief(
            bank_id, counterparty_id, belief, self.timestep, event
        )
        
        # Record for calibration
        self.evaluation_engine.record_prediction(
            bank_id, counterparty_id, 
            belief.estimated_pd, belief.confidence_score, 
            self.timestep
        )
        
        if default_occurred:
            self.evaluation_engine.record_outcome(
                counterparty_id, defaulted=True, timestep=self.timestep
            )
        
        # Update trust network
        self.trust_engine.update(self.timestep)
    
    def get_belief_observation(self,
                               bank_id: int,
                               base_observation: np.ndarray,
                               counterparty_id: Optional[int] = None) -> np.ndarray:
        """
        Get belief-augmented observation for MAPPO.
        """
        obs_builder = self.obs_builders[bank_id]
        from src.agents.belief_integration import ObservationContext
        
        belief_obs = obs_builder.build_observation(
            base_obs=base_observation,
            context=ObservationContext.COUNTERPARTY if counterparty_id else ObservationContext.GENERAL,
            target_counterparty=counterparty_id
        )
        
        return belief_obs.to_vector()
    
    def filter_action(self,
                      bank_id: int,
                      action: int,
                      counterparty_id: int) -> tuple:
        """
        Filter action through belief system.
        
        Returns:
            Tuple of (filtered_action, metadata)
        """
        belief_filter = self.belief_filters[bank_id]
        filtered_action, metadata = belief_filter.filter_action(
            raw_action=action,
            counterparty_id=counterparty_id,
            action_context={'base_interest_rate': 0.05, 'amount': 1000000}
        )
        allowed = not metadata.get('modified', False)
        reason = metadata.get('reason', 'allowed')
        return allowed, reason
    
    def get_transaction_terms(self,
                              bank_id: int,
                              counterparty_id: int,
                              amount: float = 1000000) -> Dict:
        """
        Get risk-adjusted transaction terms.
        """
        terms_engine = self.terms_engines[bank_id]
        terms = terms_engine.compute_loan_terms(counterparty_id, amount)
        
        return {
            'interest_rate': terms.interest_rate,
            'collateral_ratio': terms.collateral_ratio,
            'exposure_limit': terms.exposure_limit,
            'maturity': terms.maturity
        }
    
    def analyze_counterfactual(self,
                               bank_id: int,
                               counterparty_id: int,
                               action: str,
                               amount: float) -> Dict:
        """
        Analyze counterfactual outcomes using beliefs.
        """
        engine = self.counterfactual_engines[bank_id]
        
        result = engine.analyze_decision(
            counterparty_id=counterparty_id,
            action_type=action,
            amount=amount
        )
        
        return {
            'believed_profit': result.believed_profit,
            'believed_loss': result.believed_loss,
            'expected_value': result.believed_profit - result.believed_loss
        }
    
    def get_explanation(self,
                        bank_id: int,
                        counterparty_id: int) -> Dict:
        """
        Get explanation for a bank's belief about a counterparty.
        """
        belief = self.belief_systems[bank_id].get_belief(counterparty_id)
        explanation = self.belief_explainer.explain_belief(
            bank_id, counterparty_id, belief, self.timestep
        )
        
        return {
            'narrative': explanation.to_narrative(),
            'risk_level': explanation.risk_level,
            'recommendation': explanation.recommendation,
            'primary_reason': explanation.primary_reason,
            'factors': explanation.contributing_factors
        }
    
    def get_system_health(self) -> Dict:
        """Get system health report."""
        # Collect all beliefs
        all_beliefs = {}
        for bank_id in self.bank_ids:
            bs = self.belief_systems[bank_id]
            all_beliefs[bank_id] = {
                cp_id: b.estimated_pd 
                for cp_id, b in bs.get_all_beliefs().items()
            }
        
        return self.evaluation_engine.check_system_health(all_beliefs, self.timestep)
    
    def get_trust_network_data(self) -> Dict:
        """Get trust network for visualization."""
        return self.trust_engine.get_trust_network_data()
    
    def get_public_signals(self) -> Dict:
        """Get public signals (reputation, warnings)."""
        return self.trust_engine.get_public_signals()
    
    def export_ui_data(self) -> Dict:
        """Export all data for UI consumption."""
        exporter = BeliefUIExporter(
            belief_managers=self.belief_systems,
            trust_engine=self.trust_engine,
            belief_logger=self.belief_logger
        )
        
        return exporter.export_all_beliefs(self.timestep)
    
    def step(self) -> None:
        """Advance to next timestep."""
        self.timestep += 1
        
        # Advance all belief systems
        self.belief_manager.advance_all(self.timestep)
        
        # Update trust engine
        self.trust_engine.update(self.timestep)


def demo_belief_learning():
    """
    Demonstrate the belief learning system.
    """
    print("=" * 60)
    print("FinSim-MAPPO Belief Learning Demo")
    print("=" * 60)
    
    # Initialize system with 5 banks
    bank_ids = [0, 1, 2, 3, 4]
    system = BeliefLearningSystem(bank_ids)
    
    print("\n1. Initial State")
    print("-" * 40)
    
    # Check initial beliefs (should be at market prior)
    belief = system.get_belief_system(0).get_belief(1)
    print(f"Bank 0's belief about Bank 1:")
    print(f"  - PD: {belief.estimated_pd:.4f}")
    print(f"  - Trust: {belief.trust_score:.4f}")
    print(f"  - Confidence: {belief.confidence_score:.4f}")
    
    print("\n2. Recording Interactions")
    print("-" * 40)
    
    # Simulate some interactions
    # Bank 0 lends to Bank 1 (successful)
    system.record_interaction(
        bank_id=0,
        counterparty_id=1,
        interaction_type=InteractionType.LOAN_GIVEN,
        amount=1000000,
        rate=0.05
    )
    
    # Bank 1 repays successfully
    system.record_outcome(
        bank_id=0,
        counterparty_id=1,
        interaction_type=InteractionType.LOAN_REPAID,
        successful=True,
        actual_repayment=1050000
    )
    print("Bank 0 lent to Bank 1 - Repaid successfully")
    
    # Another successful interaction
    system.step()
    system.record_interaction(
        bank_id=0,
        counterparty_id=1,
        interaction_type=InteractionType.LOAN_GIVEN,
        amount=2000000,
        rate=0.04
    )
    system.record_outcome(
        bank_id=0,
        counterparty_id=1,
        interaction_type=InteractionType.LOAN_REPAID,
        successful=True,
        actual_repayment=2080000
    )
    print("Bank 0 lent to Bank 1 again - Repaid successfully")
    
    # Check updated beliefs
    belief = system.get_belief_system(0).get_belief(1)
    print(f"\nBank 0's updated belief about Bank 1:")
    print(f"  - PD: {belief.estimated_pd:.4f} (should decrease)")
    print(f"  - Trust: {belief.trust_score:.4f} (should increase)")
    print(f"  - Confidence: {belief.confidence_score:.4f} (should increase)")
    
    print("\n3. Simulating a Default")
    print("-" * 40)
    
    # Bank 0 lends to Bank 2 (default)
    system.record_interaction(
        bank_id=0,
        counterparty_id=2,
        interaction_type=InteractionType.LOAN_GIVEN,
        amount=500000,
        rate=0.06
    )
    system.record_outcome(
        bank_id=0,
        counterparty_id=2,
        interaction_type=InteractionType.LOAN_DEFAULTED,
        successful=False,
        default_occurred=True
    )
    print("Bank 0 lent to Bank 2 - DEFAULT occurred")
    
    belief = system.get_belief_system(0).get_belief(2)
    print(f"\nBank 0's belief about Bank 2:")
    print(f"  - PD: {belief.estimated_pd:.4f} (should increase significantly)")
    print(f"  - Trust: {belief.trust_score:.4f} (should decrease significantly)")
    
    print("\n4. Belief Heterogeneity (Information Asymmetry)")
    print("-" * 40)
    
    # Bank 3 has different experience with Bank 2
    system.record_interaction(
        bank_id=3,
        counterparty_id=2,
        interaction_type=InteractionType.LOAN_GIVEN,
        amount=300000,
        rate=0.05
    )
    system.record_outcome(
        bank_id=3,
        counterparty_id=2,
        interaction_type=InteractionType.LOAN_REPAID,
        successful=True,
        actual_repayment=315000
    )
    
    belief_0 = system.get_belief_system(0).get_belief(2)
    belief_3 = system.get_belief_system(3).get_belief(2)
    
    print(f"Bank 0's belief about Bank 2: PD={belief_0.estimated_pd:.4f}")
    print(f"Bank 3's belief about Bank 2: PD={belief_3.estimated_pd:.4f}")
    print("(Different experiences lead to different beliefs!)")
    
    print("\n5. Decision Filtering")
    print("-" * 40)
    
    # Bank 0 tries to lend to Bank 2 (should be blocked)
    allowed, reason = system.filter_action(
        bank_id=0,
        action=1,  # Assume 1 = lend
        counterparty_id=2
    )
    print(f"Bank 0 lending to Bank 2: Allowed={allowed}")
    print(f"  Reason: {reason}")
    
    # Bank 0 lending to Bank 1 (should be allowed)
    allowed, reason = system.filter_action(
        bank_id=0,
        action=1,
        counterparty_id=1
    )
    print(f"Bank 0 lending to Bank 1: Allowed={allowed}")
    
    print("\n6. Risk-Adjusted Terms")
    print("-" * 40)
    
    terms_1 = system.get_transaction_terms(0, 1)
    terms_2 = system.get_transaction_terms(0, 2)
    
    print(f"Terms for Bank 1 (low risk):")
    print(f"  - Rate: {terms_1['interest_rate']:.2%}")
    print(f"  - Collateral: {terms_1['collateral_ratio']:.0%}")
    
    print(f"\nTerms for Bank 2 (high risk):")
    print(f"  - Rate: {terms_2['interest_rate']:.2%}")
    print(f"  - Collateral: {terms_2['collateral_ratio']:.0%}")
    
    print("\n7. Explanation Generation")
    print("-" * 40)
    
    explanation = system.get_explanation(0, 2)
    print(f"Explanation for Bank 0's view of Bank 2:")
    print(f"  {explanation['narrative']}")
    print(f"  Risk Level: {explanation['risk_level']}")
    print(f"  Recommendation: {explanation['recommendation']}")
    
    print("\n8. System Health Check")
    print("-" * 40)
    
    health = system.get_system_health()
    print(f"System Health Score: {health['health_score']:.2f}")
    print(f"Herding Detected: {health['herding'].get('herding_detected', False)}")
    print(f"Trust Stability: {health['stability']['fragility_index']:.4f}")
    
    print("\n9. Trust Network")
    print("-" * 40)
    
    network = system.get_trust_network_data()
    print(f"Trust Network: {len(network['nodes'])} nodes, {len(network['edges'])} edges")
    
    print("\n10. Public Signals")
    print("-" * 40)
    
    signals = system.get_public_signals()
    print(f"System Trust Index: {signals.get('system_trust_index', 0.5):.4f}")
    
    if signals.get('high_risk_warnings'):
        print("High Risk Warnings:", signals['high_risk_warnings'])
    else:
        print("No high risk warnings")
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    
    return system


if __name__ == "__main__":
    demo_belief_learning()
