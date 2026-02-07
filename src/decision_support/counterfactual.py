"""
Counterfactual Decision-Support Engine for FinSim-MAPPO.
Provides state replication, transaction injection, and comparative analysis.
"""

import numpy as np
import copy
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import json


class TransactionType(Enum):
    """Types of hypothetical transactions."""
    LOAN_APPROVAL = "loan_approval"
    LOAN_REJECTION = "loan_rejection"
    MARGIN_INCREASE = "margin_increase"
    MARGIN_DECREASE = "margin_decrease"
    ASSET_SALE = "asset_sale"
    ASSET_PURCHASE = "asset_purchase"
    LIQUIDITY_INJECTION = "liquidity_injection"
    CAPITAL_CALL = "capital_call"
    EXPOSURE_REDUCTION = "exposure_reduction"


@dataclass
class HypotheticalTransaction:
    """Specification of a hypothetical transaction."""
    transaction_id: str
    transaction_type: TransactionType
    
    # Parties
    initiator_id: int
    counterparty_id: Optional[int] = None
    
    # Amount
    principal_amount: float = 0.0
    interest_rate: float = 0.0
    duration: int = 1  # In timesteps
    
    # Collateral
    collateral_required: float = 0.0
    collateral_type: str = "cash"
    
    # Fees
    arrangement_fee: float = 0.0
    
    # Metadata
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['transaction_type'] = self.transaction_type.value
        return result


@dataclass
class CounterfactualRiskMetrics:
    """Risk metrics computed from counterfactual simulation."""
    # Default Risk
    initiator_pd: float = 0.0  # Probability of default for initiator
    counterparty_pd: float = 0.0  # Probability of default for counterparty
    system_pd_increase: float = 0.0  # Increase in system-wide default rate
    
    # Expected Loss
    expected_credit_loss: float = 0.0
    expected_lgd: float = 0.0  # Loss given default
    
    # CCP Impact
    additional_ccp_stress: float = 0.0
    margin_call_probability: float = 0.0
    waterfall_activation_risk: float = 0.0
    
    # Liquidity
    liquidity_drain: float = 0.0
    funding_gap_increase: float = 0.0
    
    # Contagion
    contagion_depth_increase: int = 0
    affected_entities_count: int = 0
    cascade_probability: float = 0.0
    
    # Network Effects
    network_fragmentation_increase: float = 0.0
    concentration_risk_increase: float = 0.0
    
    # Summary Scores
    overall_risk_score: float = 0.0  # 0-100
    recommendation: str = ""
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def compute_overall_score(self) -> float:
        """Compute overall risk score from components."""
        # Weighted combination
        score = (
            self.initiator_pd * 30 +
            self.counterparty_pd * 20 +
            self.cascade_probability * 25 +
            min(self.liquidity_drain / 1e6, 1) * 15 +
            min(self.additional_ccp_stress / 1e6, 1) * 10
        )
        self.overall_risk_score = min(100, max(0, score * 100))
        return self.overall_risk_score


@dataclass
class CounterfactualResult:
    """Complete result of a counterfactual analysis."""
    # Identification
    analysis_id: str
    timestamp: str
    
    # Transaction
    transaction: HypotheticalTransaction
    
    # Baseline outcome (without transaction)
    baseline_initiator_equity: float = 0.0
    baseline_initiator_capital_ratio: float = 0.0
    baseline_system_defaults: int = 0
    baseline_system_stress: int = 0
    baseline_total_exposure: float = 0.0
    
    # Counterfactual outcome (with transaction)
    cf_initiator_equity: float = 0.0
    cf_initiator_capital_ratio: float = 0.0
    cf_system_defaults: int = 0
    cf_system_stress: int = 0
    cf_total_exposure: float = 0.0
    
    # Deltas
    equity_change: float = 0.0
    capital_ratio_change: float = 0.0
    defaults_change: int = 0
    exposure_change: float = 0.0
    
    # Risk metrics
    risk_metrics: CounterfactualRiskMetrics = field(default_factory=CounterfactualRiskMetrics)
    
    # Simulation details
    horizon: int = 10
    num_simulations: int = 20
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['transaction'] = self.transaction.to_dict()
        result['risk_metrics'] = self.risk_metrics.to_dict()
        return result


class StateReplicator:
    """
    Handles complete replication of system state for counterfactual analysis.
    """
    
    @staticmethod
    def replicate_bank(bank: Any) -> Any:
        """Create deep copy of bank state."""
        return copy.deepcopy(bank)
    
    @staticmethod
    def replicate_exchange(exchange: Any) -> Any:
        """Create deep copy of exchange state."""
        return copy.deepcopy(exchange)
    
    @staticmethod
    def replicate_ccp(ccp: Any) -> Any:
        """Create deep copy of CCP state."""
        return copy.deepcopy(ccp)
    
    @staticmethod
    def replicate_market(market: Any) -> Any:
        """Create deep copy of market state."""
        return copy.deepcopy(market)
    
    @staticmethod
    def replicate_network(network: Any) -> Any:
        """Create deep copy of network including all banks."""
        return copy.deepcopy(network)
    
    @staticmethod
    def replicate_environment(env: Any) -> Any:
        """Create complete deep copy of environment."""
        return copy.deepcopy(env)
    
    @staticmethod
    def replicate_full_state(
        env: Any,
        exchanges: Optional[List[Any]] = None,
        ccps: Optional[List[Any]] = None
    ) -> Tuple[Any, List[Any], List[Any]]:
        """
        Replicate complete system state.
        
        Returns:
            Tuple of (env_copy, exchanges_copy, ccps_copy)
        """
        env_copy = copy.deepcopy(env)
        exchanges_copy = [copy.deepcopy(e) for e in (exchanges or [])]
        ccps_copy = [copy.deepcopy(c) for c in (ccps or [])]
        
        return env_copy, exchanges_copy, ccps_copy


class TransactionInjector:
    """
    Injects hypothetical transactions into replicated environment.
    """
    
    def __init__(self):
        self.injection_log: List[Dict[str, Any]] = []
    
    def inject_loan(
        self,
        env: Any,
        lender_id: int,
        borrower_id: int,
        principal: float,
        interest_rate: float = 0.05,
        ccps: Optional[List[Any]] = None
    ) -> bool:
        """
        Inject a loan transaction into the environment.
        
        Adjusts:
        - Lender: cash down, interbank_assets up
        - Borrower: cash up, interbank_liabilities up
        - Network: liability matrix updated
        - CCP: exposure updated if applicable
        """
        lender = env.network.banks.get(lender_id)
        borrower = env.network.banks.get(borrower_id)
        
        if not lender or not borrower:
            return False
        
        # Check if lender has sufficient liquidity
        if lender.balance_sheet.cash < principal:
            return False
        
        # Adjust lender balance sheet
        lender.balance_sheet.cash -= principal
        lender.balance_sheet.interbank_assets += principal
        
        # Adjust borrower balance sheet
        borrower.balance_sheet.cash += principal
        borrower.balance_sheet.interbank_liabilities += principal
        
        # Update liability matrix
        if hasattr(env.network, 'liability_matrix'):
            env.network.liability_matrix[borrower_id, lender_id] += principal
        
        # Update CCP exposures if applicable
        if ccps:
            for ccp in ccps:
                if hasattr(ccp, 'update_exposure'):
                    ccp.update_exposure(lender_id, borrower_id, principal)
        
        # Log injection
        self.injection_log.append({
            'type': 'loan',
            'lender_id': lender_id,
            'borrower_id': borrower_id,
            'principal': principal,
            'interest_rate': interest_rate
        })
        
        return True
    
    def inject_margin_change(
        self,
        ccp: Any,
        member_id: int,
        margin_change: float
    ) -> bool:
        """
        Inject a margin change at a CCP.
        
        Positive = increase required margin
        Negative = release margin
        """
        if not hasattr(ccp, 'margin_accounts'):
            return False
        
        if member_id not in ccp.margin_accounts:
            return False
        
        account = ccp.margin_accounts[member_id]
        account.initial_margin += margin_change
        
        self.injection_log.append({
            'type': 'margin_change',
            'ccp_id': ccp.ccp_id,
            'member_id': member_id,
            'change': margin_change
        })
        
        return True
    
    def inject_asset_sale(
        self,
        env: Any,
        bank_id: int,
        amount: float,
        price_impact: float = 0.01
    ) -> bool:
        """
        Inject an asset sale into the environment.
        
        Adjusts:
        - Bank: securities down, cash up (with price impact)
        - Market: price potentially affected
        """
        bank = env.network.banks.get(bank_id)
        if not bank:
            return False
        
        if bank.balance_sheet.securities < amount:
            return False
        
        # Calculate sale proceeds with price impact
        proceeds = amount * (1 - price_impact)
        
        # Adjust balance sheet
        bank.balance_sheet.securities -= amount
        bank.balance_sheet.cash += proceeds
        
        # Impact market price
        if hasattr(env, 'market'):
            env.market.apply_price_impact(-amount * price_impact)
        
        self.injection_log.append({
            'type': 'asset_sale',
            'bank_id': bank_id,
            'amount': amount,
            'proceeds': proceeds,
            'price_impact': price_impact
        })
        
        return True
    
    def inject_liquidity(
        self,
        env: Any,
        bank_id: int,
        amount: float,
        source: str = "central_bank"
    ) -> bool:
        """
        Inject liquidity into a bank.
        """
        bank = env.network.banks.get(bank_id)
        if not bank:
            return False
        
        bank.balance_sheet.cash += amount
        
        if source == "central_bank":
            # Create liability to central bank
            bank.balance_sheet.deposits += amount  # Simplified
        
        self.injection_log.append({
            'type': 'liquidity_injection',
            'bank_id': bank_id,
            'amount': amount,
            'source': source
        })
        
        return True
    
    def inject_transaction(
        self,
        transaction: HypotheticalTransaction,
        env: Any,
        exchanges: Optional[List[Any]] = None,
        ccps: Optional[List[Any]] = None
    ) -> bool:
        """
        Inject a hypothetical transaction based on specification.
        """
        if transaction.transaction_type == TransactionType.LOAN_APPROVAL:
            return self.inject_loan(
                env,
                transaction.initiator_id,
                transaction.counterparty_id,
                transaction.principal_amount,
                transaction.interest_rate,
                ccps
            )
        
        elif transaction.transaction_type == TransactionType.ASSET_SALE:
            return self.inject_asset_sale(
                env,
                transaction.initiator_id,
                transaction.principal_amount
            )
        
        elif transaction.transaction_type == TransactionType.LIQUIDITY_INJECTION:
            return self.inject_liquidity(
                env,
                transaction.initiator_id,
                transaction.principal_amount
            )
        
        elif transaction.transaction_type == TransactionType.MARGIN_INCREASE:
            if ccps and transaction.counterparty_id is not None:
                for ccp in ccps:
                    if ccp.ccp_id == transaction.counterparty_id:
                        return self.inject_margin_change(
                            ccp,
                            transaction.initiator_id,
                            transaction.principal_amount
                        )
        
        return False
    
    def clear_log(self) -> None:
        """Clear injection log."""
        self.injection_log.clear()


class CounterfactualEngine:
    """
    Main engine for counterfactual decision support.
    
    Provides:
    - State replication
    - Transaction injection
    - Forward simulation
    - Comparative analysis
    - Risk metric computation
    """
    
    def __init__(
        self,
        horizon: int = 10,
        num_simulations: int = 20,
        seed: Optional[int] = None
    ):
        self.horizon = horizon
        self.num_simulations = num_simulations
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        
        self.replicator = StateReplicator()
        self.injector = TransactionInjector()
        
        self._analysis_counter = 0
    
    def _next_analysis_id(self) -> str:
        """Generate unique analysis ID."""
        self._analysis_counter += 1
        return f"CFA_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._analysis_counter:04d}"
    
    def analyze_transaction(
        self,
        transaction: HypotheticalTransaction,
        env: Any,
        agents: Dict[int, Any],
        exchanges: Optional[List[Any]] = None,
        ccps: Optional[List[Any]] = None,
        observations: Optional[Dict[int, np.ndarray]] = None
    ) -> CounterfactualResult:
        """
        Analyze a hypothetical transaction by comparing baseline vs counterfactual.
        
        Args:
            transaction: The hypothetical transaction to analyze
            env: Current environment state
            agents: Agent policies for forward simulation
            exchanges: Optional list of exchanges
            ccps: Optional list of CCPs
            observations: Current observations (if available)
            
        Returns:
            Complete counterfactual analysis result
        """
        # Get initial observations if not provided
        if observations is None:
            observations = {i: np.zeros(16) for i in range(len(env.network.banks))}
        
        # Run baseline simulation
        baseline_results = self._simulate_forward(
            env, agents, exchanges, ccps, observations,
            inject_transaction=False
        )
        
        # Run counterfactual simulation
        cf_results = self._simulate_forward(
            env, agents, exchanges, ccps, observations,
            inject_transaction=True,
            transaction=transaction
        )
        
        # Compute risk metrics
        risk_metrics = self._compute_risk_metrics(
            transaction, baseline_results, cf_results
        )
        
        # Build result
        result = CounterfactualResult(
            analysis_id=self._next_analysis_id(),
            timestamp=datetime.now().isoformat(),
            transaction=transaction,
            baseline_initiator_equity=baseline_results['initiator_equity'],
            baseline_initiator_capital_ratio=baseline_results['initiator_capital_ratio'],
            baseline_system_defaults=baseline_results['system_defaults'],
            baseline_system_stress=baseline_results['system_stress'],
            baseline_total_exposure=baseline_results['total_exposure'],
            cf_initiator_equity=cf_results['initiator_equity'],
            cf_initiator_capital_ratio=cf_results['initiator_capital_ratio'],
            cf_system_defaults=cf_results['system_defaults'],
            cf_system_stress=cf_results['system_stress'],
            cf_total_exposure=cf_results['total_exposure'],
            equity_change=cf_results['initiator_equity'] - baseline_results['initiator_equity'],
            capital_ratio_change=cf_results['initiator_capital_ratio'] - baseline_results['initiator_capital_ratio'],
            defaults_change=cf_results['system_defaults'] - baseline_results['system_defaults'],
            exposure_change=cf_results['total_exposure'] - baseline_results['total_exposure'],
            risk_metrics=risk_metrics,
            horizon=self.horizon,
            num_simulations=self.num_simulations
        )
        
        return result
    
    def _simulate_forward(
        self,
        env: Any,
        agents: Dict[int, Any],
        exchanges: Optional[List[Any]],
        ccps: Optional[List[Any]],
        observations: Dict[int, np.ndarray],
        inject_transaction: bool = False,
        transaction: Optional[HypotheticalTransaction] = None
    ) -> Dict[str, Any]:
        """
        Run forward simulation with optional transaction injection.
        """
        # Aggregate results across simulations
        initiator_equities = []
        initiator_capital_ratios = []
        initiator_defaults = 0
        counterparty_defaults = 0
        system_defaults = []
        system_stress = []
        total_exposures = []
        
        initiator_id = transaction.initiator_id if transaction else 0
        counterparty_id = transaction.counterparty_id if transaction else None
        
        for sim in range(self.num_simulations):
            # Replicate state
            env_copy, ex_copy, ccp_copy = self.replicator.replicate_full_state(
                env, exchanges, ccps
            )
            
            # Inject transaction if specified
            if inject_transaction and transaction:
                self.injector.inject_transaction(
                    transaction, env_copy, ex_copy, ccp_copy
                )
            
            # Get fresh observations
            obs = {k: v.copy() for k, v in observations.items()}
            
            # Simulate forward
            for step in range(self.horizon):
                # Get actions from agents
                actions = {}
                for agent_id, agent in agents.items():
                    if hasattr(agent, 'select_action'):
                        actions[agent_id] = agent.select_action(
                            obs.get(agent_id, np.zeros(16)),
                            deterministic=True
                        )
                    else:
                        actions[agent_id] = np.zeros(4)
                
                # Step environment
                try:
                    result = env_copy.step(actions)
                    obs = result.observations
                    
                    if all(result.dones.values()):
                        break
                except Exception:
                    break
            
            # Collect final state metrics
            initiator_bank = env_copy.network.banks.get(initiator_id)
            if initiator_bank:
                initiator_equities.append(initiator_bank.balance_sheet.equity)
                initiator_capital_ratios.append(initiator_bank.capital_ratio)
                if initiator_bank.status.value == 'defaulted':
                    initiator_defaults += 1
            
            if counterparty_id is not None:
                cp_bank = env_copy.network.banks.get(counterparty_id)
                if cp_bank and cp_bank.status.value == 'defaulted':
                    counterparty_defaults += 1
            
            stats = env_copy.network.get_network_stats()
            system_defaults.append(stats.num_defaulted)
            system_stress.append(stats.num_stressed)
            total_exposures.append(stats.total_exposure)
        
        n = max(self.num_simulations, 1)
        
        return {
            'initiator_equity': np.mean(initiator_equities) if initiator_equities else 0,
            'initiator_capital_ratio': np.mean(initiator_capital_ratios) if initiator_capital_ratios else 0,
            'initiator_pd': initiator_defaults / n,
            'counterparty_pd': counterparty_defaults / n if counterparty_id else 0,
            'system_defaults': int(np.mean(system_defaults)) if system_defaults else 0,
            'system_stress': int(np.mean(system_stress)) if system_stress else 0,
            'total_exposure': np.mean(total_exposures) if total_exposures else 0,
            'defaults_std': np.std(system_defaults) if system_defaults else 0,
        }
    
    def _compute_risk_metrics(
        self,
        transaction: HypotheticalTransaction,
        baseline: Dict[str, Any],
        counterfactual: Dict[str, Any]
    ) -> CounterfactualRiskMetrics:
        """
        Compute comprehensive risk metrics from simulation results.
        """
        metrics = CounterfactualRiskMetrics(
            initiator_pd=counterfactual['initiator_pd'],
            counterparty_pd=counterfactual['counterparty_pd'],
            system_pd_increase=max(0, counterfactual['system_defaults'] - baseline['system_defaults']) / 20,
            expected_credit_loss=transaction.principal_amount * counterfactual['counterparty_pd'] * 0.45,
            expected_lgd=0.45,  # Standard LGD assumption
            liquidity_drain=transaction.principal_amount if transaction.transaction_type == TransactionType.LOAN_APPROVAL else 0,
            contagion_depth_increase=max(0, counterfactual['system_defaults'] - baseline['system_defaults']),
            affected_entities_count=counterfactual['system_defaults'],
            cascade_probability=min(1.0, counterfactual['defaults_std'] / 5) if counterfactual['defaults_std'] > 0 else 0,
        )
        
        # Compute overall score
        metrics.compute_overall_score()
        
        # Generate recommendation
        if metrics.overall_risk_score < 20:
            metrics.recommendation = "APPROVE - Low risk transaction"
            metrics.confidence = 0.9
        elif metrics.overall_risk_score < 40:
            metrics.recommendation = "APPROVE WITH CONDITIONS - Monitor closely"
            metrics.confidence = 0.7
        elif metrics.overall_risk_score < 60:
            metrics.recommendation = "REVIEW REQUIRED - Moderate risk"
            metrics.confidence = 0.6
        elif metrics.overall_risk_score < 80:
            metrics.recommendation = "CAUTION - High risk, additional collateral recommended"
            metrics.confidence = 0.7
        else:
            metrics.recommendation = "REJECT - Excessive systemic risk"
            metrics.confidence = 0.85
        
        return metrics
    
    def analyze_loan_approval(
        self,
        lender_id: int,
        borrower_id: int,
        principal: float,
        interest_rate: float,
        env: Any,
        agents: Dict[int, Any],
        **kwargs
    ) -> CounterfactualResult:
        """
        Convenience method for loan approval analysis.
        """
        transaction = HypotheticalTransaction(
            transaction_id=f"LOAN_{lender_id}_{borrower_id}_{int(principal)}",
            transaction_type=TransactionType.LOAN_APPROVAL,
            initiator_id=lender_id,
            counterparty_id=borrower_id,
            principal_amount=principal,
            interest_rate=interest_rate,
            description=f"Loan from Bank {lender_id} to Bank {borrower_id}"
        )
        
        return self.analyze_transaction(transaction, env, agents, **kwargs)
    
    def compare_scenarios(
        self,
        transactions: List[HypotheticalTransaction],
        env: Any,
        agents: Dict[int, Any],
        **kwargs
    ) -> List[CounterfactualResult]:
        """
        Compare multiple transaction scenarios.
        """
        results = []
        for transaction in transactions:
            result = self.analyze_transaction(transaction, env, agents, **kwargs)
            results.append(result)
        
        # Rank by risk score
        results.sort(key=lambda r: r.risk_metrics.overall_risk_score)
        
        return results
    
    def generate_report(self, result: CounterfactualResult) -> str:
        """
        Generate human-readable report from analysis result.
        """
        t = result.transaction
        m = result.risk_metrics
        
        report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    COUNTERFACTUAL ANALYSIS REPORT                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Analysis ID: {result.analysis_id:<60s} ║
║ Timestamp:   {result.timestamp:<60s} ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ TRANSACTION DETAILS                                                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Type:        {t.transaction_type.value:<60s} ║
║ Initiator:   Bank {t.initiator_id:<55d} ║
║ Counterparty: Bank {t.counterparty_id if t.counterparty_id else 'N/A'!s:<54s} ║
║ Principal:   ${t.principal_amount:,.2f}{' '*(50-len(f'{t.principal_amount:,.2f}'))}║
╠══════════════════════════════════════════════════════════════════════════════╣
║ IMPACT ANALYSIS                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                          Baseline        With Transaction        Delta       ║
║ Initiator Equity:    ${result.baseline_initiator_equity:>12,.0f}      ${result.cf_initiator_equity:>12,.0f}    {result.equity_change:>+12,.0f} ║
║ Capital Ratio:           {result.baseline_initiator_capital_ratio*100:>8.2f}%          {result.cf_initiator_capital_ratio*100:>8.2f}%   {result.capital_ratio_change*100:>+8.2f}% ║
║ System Defaults:            {result.baseline_system_defaults:>5d}                 {result.cf_system_defaults:>5d}        {result.defaults_change:>+5d} ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ RISK METRICS                                                                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Initiator Default Probability:     {m.initiator_pd*100:>6.2f}%                              ║
║ Counterparty Default Probability:  {m.counterparty_pd*100:>6.2f}%                              ║
║ Expected Credit Loss:              ${m.expected_credit_loss:>12,.0f}                       ║
║ Cascade Probability:               {m.cascade_probability*100:>6.2f}%                              ║
║ Liquidity Drain:                   ${m.liquidity_drain:>12,.0f}                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ OVERALL RISK SCORE:                {m.overall_risk_score:>6.1f} / 100                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ RECOMMENDATION: {m.recommendation:<56s} ║
║ Confidence:     {m.confidence*100:.0f}%                                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        return report
