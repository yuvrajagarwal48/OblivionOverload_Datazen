"""
Risk Metrics: Systemic risk analysis and financial stability indicators.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import networkx as nx


@dataclass
class RiskReport:
    """Comprehensive risk report for the financial network."""
    # Overall metrics
    debt_rank: float
    systemic_risk_index: float
    liquidity_index: float
    stress_index: float
    
    # Network metrics
    network_density: float
    avg_clustering: float
    largest_component_size: float
    
    # Bank-level metrics
    bank_risk_scores: Dict[int, float]
    systemically_important_banks: List[int]
    vulnerable_banks: List[int]
    
    # Contagion metrics
    contagion_depth: float
    cascade_potential: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'debt_rank': self.debt_rank,
            'systemic_risk_index': self.systemic_risk_index,
            'liquidity_index': self.liquidity_index,
            'stress_index': self.stress_index,
            'network_density': self.network_density,
            'avg_clustering': self.avg_clustering,
            'largest_component_size': self.largest_component_size,
            'bank_risk_scores': self.bank_risk_scores,
            'systemically_important_banks': self.systemically_important_banks,
            'vulnerable_banks': self.vulnerable_banks,
            'contagion_depth': self.contagion_depth,
            'cascade_potential': self.cascade_potential
        }


class DebtRankCalculator:
    """
    Calculates DebtRank - a measure of systemic importance.
    
    DebtRank measures the fraction of total economic value in the network
    that is potentially affected by the distress of a single institution.
    """
    
    def __init__(self, 
                 recovery_rate: float = 0.0,
                 max_iterations: int = 100,
                 threshold: float = 1e-6):
        """
        Initialize DebtRank calculator.
        
        Args:
            recovery_rate: Expected recovery in case of default
            max_iterations: Maximum iterations for propagation
            threshold: Convergence threshold
        """
        self.recovery_rate = recovery_rate
        self.max_iterations = max_iterations
        self.threshold = threshold
    
    def calculate(self,
                  exposure_matrix: np.ndarray,
                  equity_vector: np.ndarray,
                  initial_shocks: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
        """
        Calculate DebtRank for the network.
        
        Args:
            exposure_matrix: Matrix of exposures (i,j) = exposure of i to j
            equity_vector: Equity of each bank
            initial_shocks: Initial shock to each bank (default: shock each bank individually)
            
        Returns:
            Tuple of (aggregate_debtrank, individual_debtranks)
        """
        n = len(equity_vector)
        
        # Build impact matrix W where W[i,j] = min(1, exposure[i,j] / equity[i])
        impact_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if equity_vector[i] > 0:
                    impact_matrix[i, j] = min(1.0, exposure_matrix[i, j] / equity_vector[i])
        
        # Calculate economic value (proportional to equity)
        total_equity = np.sum(equity_vector)
        if total_equity <= 0:
            return 0.0, np.zeros(n)
        
        value_weights = equity_vector / total_equity
        
        # Calculate DebtRank for each bank as initial shock source
        individual_debtranks = np.zeros(n)
        
        for source in range(n):
            if equity_vector[source] <= 0:
                continue
            
            # Initialize distress levels
            h = np.zeros(n)  # Cumulative distress
            s = np.zeros(n)  # Active distress
            
            # Initial shock
            if initial_shocks is not None:
                s[source] = initial_shocks[source]
            else:
                s[source] = 1.0  # Full default
            
            h[source] = s[source]
            
            # Propagation
            for _ in range(self.max_iterations):
                # Calculate new distress
                s_new = np.zeros(n)
                
                for i in range(n):
                    if h[i] >= 1.0:
                        continue
                    
                    # Impact from neighbors
                    impact = 0.0
                    for j in range(n):
                        if s[j] > 0:
                            impact += impact_matrix[i, j] * s[j] * (1 - self.recovery_rate)
                    
                    s_new[i] = min(1.0 - h[i], impact)
                
                # Update cumulative distress
                h = np.minimum(1.0, h + s_new)
                
                # Check convergence
                if np.max(s_new) < self.threshold:
                    break
                
                s = s_new
            
            # DebtRank is value-weighted sum of distress minus initial shock
            debtrank = np.sum(value_weights * h) - value_weights[source]
            individual_debtranks[source] = max(0.0, debtrank)
        
        # Aggregate DebtRank (average over all possible initial shocks)
        aggregate_debtrank = np.mean(individual_debtranks)
        
        return aggregate_debtrank, individual_debtranks
    
    def calculate_for_bank(self,
                           bank_id: int,
                           exposure_matrix: np.ndarray,
                           equity_vector: np.ndarray) -> float:
        """Calculate DebtRank contribution of a single bank."""
        initial_shocks = np.zeros(len(equity_vector))
        initial_shocks[bank_id] = 1.0
        
        _, individual = self.calculate(exposure_matrix, equity_vector, initial_shocks)
        return individual[bank_id]


class ContagionAnalyzer:
    """
    Analyzes contagion dynamics in the financial network.
    """
    
    def __init__(self, max_rounds: int = 50):
        """
        Initialize contagion analyzer.
        
        Args:
            max_rounds: Maximum rounds of contagion to simulate
        """
        self.max_rounds = max_rounds
    
    def simulate_cascade(self,
                         exposure_matrix: np.ndarray,
                         equity_vector: np.ndarray,
                         initial_defaults: List[int]) -> Dict[str, Any]:
        """
        Simulate a default cascade from initial failures.
        
        Args:
            exposure_matrix: Matrix of exposures
            equity_vector: Equity of each bank
            initial_defaults: List of initially defaulting banks
            
        Returns:
            Cascade analysis results
        """
        n = len(equity_vector)
        
        # Track state
        defaulted = set(initial_defaults)
        equity = equity_vector.copy()
        
        cascade_rounds = []
        current_round = list(initial_defaults)
        cascade_rounds.append(current_round)
        
        # Simulate cascade
        for round_num in range(self.max_rounds):
            new_defaults = []
            
            for bank_id in current_round:
                if bank_id not in defaulted:
                    continue
                
                # Apply losses to creditors
                for creditor in range(n):
                    if creditor in defaulted:
                        continue
                    
                    loss = exposure_matrix[creditor, bank_id]
                    equity[creditor] -= loss
                    
                    if equity[creditor] <= 0:
                        new_defaults.append(creditor)
                        defaulted.add(creditor)
            
            if not new_defaults:
                break
            
            cascade_rounds.append(new_defaults)
            current_round = new_defaults
        
        return {
            'total_defaults': len(defaulted),
            'cascade_depth': len(cascade_rounds),
            'cascade_rounds': cascade_rounds,
            'final_defaults': list(defaulted),
            'default_sequence': [bank for round_banks in cascade_rounds for bank in round_banks]
        }
    
    def find_critical_sets(self,
                           exposure_matrix: np.ndarray,
                           equity_vector: np.ndarray,
                           max_set_size: int = 3) -> List[Tuple[List[int], int]]:
        """
        Find sets of banks whose simultaneous failure causes maximum damage.
        
        Args:
            exposure_matrix: Matrix of exposures
            equity_vector: Equity of each bank
            max_set_size: Maximum size of failure sets to consider
            
        Returns:
            List of (bank_set, resulting_defaults) sorted by impact
        """
        from itertools import combinations
        
        n = len(equity_vector)
        results = []
        
        for set_size in range(1, min(max_set_size + 1, n + 1)):
            for bank_set in combinations(range(n), set_size):
                cascade_result = self.simulate_cascade(
                    exposure_matrix, equity_vector, list(bank_set)
                )
                results.append((list(bank_set), cascade_result['total_defaults']))
        
        # Sort by number of resulting defaults
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:20]  # Top 20 most dangerous sets


class SystemicRiskMetrics:
    """
    Computes various systemic risk metrics.
    """
    
    @staticmethod
    def calculate_liquidity_index(cash_vector: np.ndarray,
                                   liability_vector: np.ndarray) -> float:
        """
        Calculate system-wide liquidity index.
        
        Returns value between 0 (illiquid) and 1 (highly liquid).
        """
        total_cash = np.sum(cash_vector)
        total_liabilities = np.sum(liability_vector)
        
        if total_liabilities <= 0:
            return 1.0
        
        return min(1.0, total_cash / total_liabilities)
    
    @staticmethod
    def calculate_stress_index(capital_ratios: np.ndarray,
                                min_capital_ratio: float = 0.08) -> float:
        """
        Calculate system stress index based on capital ratios.
        
        Returns value between 0 (no stress) and 1 (severe stress).
        """
        # Count banks below minimum
        below_min = np.sum(capital_ratios < min_capital_ratio)
        stressed = np.sum((capital_ratios >= min_capital_ratio) & 
                         (capital_ratios < min_capital_ratio * 1.5))
        
        n = len(capital_ratios)
        if n == 0:
            return 0.0
        
        # Weighted stress
        stress = (below_min * 1.0 + stressed * 0.5) / n
        return min(1.0, stress)
    
    @staticmethod
    def calculate_concentration_index(exposure_matrix: np.ndarray) -> float:
        """
        Calculate exposure concentration (Herfindahl-like index).
        
        Higher values indicate more concentrated exposures.
        """
        total_exposure = np.sum(exposure_matrix)
        if total_exposure <= 0:
            return 0.0
        
        # Calculate each bank's share of total exposure
        bank_exposures = np.sum(exposure_matrix, axis=1)
        shares = bank_exposures / total_exposure
        
        # Herfindahl index
        hhi = np.sum(shares ** 2)
        
        return hhi
    
    @staticmethod
    def calculate_interconnectedness(graph: nx.DiGraph) -> Dict[str, float]:
        """
        Calculate network interconnectedness metrics.
        """
        n = graph.number_of_nodes()
        
        if n == 0:
            return {'density': 0.0, 'clustering': 0.0, 'avg_path_length': 0.0}
        
        density = nx.density(graph)
        
        try:
            clustering = nx.average_clustering(graph.to_undirected())
        except:
            clustering = 0.0
        
        try:
            # Average shortest path (for largest connected component)
            if nx.is_weakly_connected(graph):
                avg_path = nx.average_shortest_path_length(graph)
            else:
                largest_cc = max(nx.weakly_connected_components(graph), key=len)
                subgraph = graph.subgraph(largest_cc)
                avg_path = nx.average_shortest_path_length(subgraph)
        except:
            avg_path = float('inf')
        
        return {
            'density': density,
            'clustering': clustering,
            'avg_path_length': avg_path
        }


class RiskAnalyzer:
    """
    Comprehensive risk analysis for the financial network.
    """
    
    def __init__(self):
        """Initialize risk analyzer."""
        self.debt_rank_calc = DebtRankCalculator()
        self.contagion_analyzer = ContagionAnalyzer()
    
    def analyze(self,
                exposure_matrix: np.ndarray,
                equity_vector: np.ndarray,
                cash_vector: np.ndarray,
                liability_vector: np.ndarray,
                capital_ratios: np.ndarray,
                graph: nx.DiGraph,
                min_capital_ratio: float = 0.08) -> RiskReport:
        """
        Perform comprehensive risk analysis.
        
        Args:
            exposure_matrix: Interbank exposure matrix
            equity_vector: Equity of each bank
            cash_vector: Cash holdings
            liability_vector: Total liabilities
            capital_ratios: Capital ratio of each bank
            graph: Network graph
            min_capital_ratio: Minimum capital ratio requirement
            
        Returns:
            Comprehensive risk report
        """
        n = len(equity_vector)
        
        # Calculate DebtRank
        aggregate_debtrank, individual_debtranks = self.debt_rank_calc.calculate(
            exposure_matrix, equity_vector
        )
        
        # Systemic risk metrics
        liquidity_index = SystemicRiskMetrics.calculate_liquidity_index(
            cash_vector, liability_vector
        )
        
        stress_index = SystemicRiskMetrics.calculate_stress_index(
            capital_ratios, min_capital_ratio
        )
        
        # Network metrics
        interconnectedness = SystemicRiskMetrics.calculate_interconnectedness(graph)
        
        # Find systemically important banks (top DebtRank contributors)
        debtrank_threshold = np.percentile(individual_debtranks, 80)
        sib_list = [i for i in range(n) if individual_debtranks[i] >= debtrank_threshold]
        
        # Find vulnerable banks (low capital ratio)
        vulnerable = [i for i in range(n) if capital_ratios[i] < min_capital_ratio * 1.5]
        
        # Calculate bank risk scores (combination of vulnerability and systemic importance)
        bank_risk_scores = {}
        for i in range(n):
            vulnerability = 1.0 - min(capital_ratios[i] / min_capital_ratio, 2.0) / 2.0
            systemic_importance = individual_debtranks[i] / max(individual_debtranks.max(), 1e-6)
            bank_risk_scores[i] = 0.5 * vulnerability + 0.5 * systemic_importance
        
        # Contagion analysis
        if sib_list:
            cascade_result = self.contagion_analyzer.simulate_cascade(
                exposure_matrix, equity_vector, [sib_list[0]]
            )
            contagion_depth = cascade_result['cascade_depth']
            cascade_potential = cascade_result['total_defaults'] / n
        else:
            contagion_depth = 0
            cascade_potential = 0.0
        
        # Systemic risk index (composite)
        systemic_risk_index = (
            0.3 * aggregate_debtrank +
            0.2 * (1 - liquidity_index) +
            0.2 * stress_index +
            0.15 * interconnectedness['density'] +
            0.15 * cascade_potential
        )
        
        # Get largest connected component
        try:
            largest_cc = max(nx.weakly_connected_components(graph), key=len)
            largest_component_size = len(largest_cc) / n
        except:
            largest_component_size = 1.0
        
        return RiskReport(
            debt_rank=aggregate_debtrank,
            systemic_risk_index=systemic_risk_index,
            liquidity_index=liquidity_index,
            stress_index=stress_index,
            network_density=interconnectedness['density'],
            avg_clustering=interconnectedness['clustering'],
            largest_component_size=largest_component_size,
            bank_risk_scores=bank_risk_scores,
            systemically_important_banks=sib_list,
            vulnerable_banks=vulnerable,
            contagion_depth=contagion_depth,
            cascade_potential=cascade_potential
        )
    
    def quick_assessment(self,
                         default_rate: float,
                         avg_capital_ratio: float,
                         liquidity_index: float) -> str:
        """
        Quick assessment of system health.
        
        Returns: 'healthy', 'caution', 'warning', 'critical'
        """
        if default_rate > 0.3 or avg_capital_ratio < 0.05:
            return 'critical'
        elif default_rate > 0.15 or avg_capital_ratio < 0.08 or liquidity_index < 0.3:
            return 'warning'
        elif default_rate > 0.05 or avg_capital_ratio < 0.12 or liquidity_index < 0.5:
            return 'caution'
        else:
            return 'healthy'
