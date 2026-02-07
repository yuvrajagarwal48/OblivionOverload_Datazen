"""
Clearing Mechanism: Eisenberg-Noe algorithm for systemic solvency resolution.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ClearingResult:
    """Results from the clearing mechanism."""
    payment_vector: np.ndarray  # Actual payments made by each bank
    default_set: List[int]  # IDs of defaulted banks
    recovery_rates: np.ndarray  # Recovery rate for each bank
    total_defaults: int
    total_losses: float
    iterations: int
    converged: bool
    
    def to_dict(self) -> dict:
        return {
            'payment_vector': self.payment_vector.tolist(),
            'default_set': self.default_set,
            'recovery_rates': self.recovery_rates.tolist(),
            'total_defaults': self.total_defaults,
            'total_losses': self.total_losses,
            'iterations': self.iterations,
            'converged': self.converged
        }


class ClearingMechanism:
    """
    Implements the Eisenberg-Noe fixed-point algorithm for clearing payments
    in an interbank network with potential defaults.
    
    The algorithm finds a consistent payment vector where each bank pays
    the minimum of its obligations and available resources.
    """
    
    def __init__(self,
                 max_iterations: int = 100,
                 convergence_threshold: float = 1e-6,
                 recovery_rate_floor: float = 0.0):
        """
        Initialize the clearing mechanism.
        
        Args:
            max_iterations: Maximum iterations for fixed-point algorithm
            convergence_threshold: Convergence criterion for payment vector
            recovery_rate_floor: Minimum recovery rate (0 = can lose everything)
        """
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.recovery_rate_floor = recovery_rate_floor
    
    def clear(self,
              liability_matrix: np.ndarray,
              external_assets: np.ndarray,
              external_liabilities: np.ndarray) -> ClearingResult:
        """
        Run the Eisenberg-Noe clearing algorithm.
        
        Args:
            liability_matrix: L[i,j] = amount bank i owes to bank j
            external_assets: Cash and illiquid assets for each bank
            external_liabilities: External (non-interbank) liabilities
            
        Returns:
            ClearingResult with payment vector, defaults, and recovery rates
        """
        n = len(external_assets)
        
        # Total obligations (sum of each row in liability matrix)
        total_obligations = np.sum(liability_matrix, axis=1)
        
        # Relative liability matrix (fraction of obligation to each creditor)
        # Pi[i,j] = L[i,j] / sum_k(L[i,k])
        with np.errstate(divide='ignore', invalid='ignore'):
            relative_liabilities = np.where(
                total_obligations[:, np.newaxis] > 0,
                liability_matrix / total_obligations[:, np.newaxis],
                0.0
            )
        
        # Initialize payment vector (assume full payment)
        payment_vector = total_obligations.copy()
        
        # Fixed-point iteration
        converged = False
        iteration = 0
        
        for iteration in range(self.max_iterations):
            prev_payment = payment_vector.copy()
            
            # Calculate available resources for each bank
            # Resources = external assets + payments received - external liabilities
            payments_received = relative_liabilities.T @ payment_vector
            available_resources = external_assets + payments_received - external_liabilities
            
            # Each bank pays minimum of obligations and available resources
            payment_vector = np.minimum(total_obligations, np.maximum(0, available_resources))
            
            # Check convergence
            if np.max(np.abs(payment_vector - prev_payment)) < self.convergence_threshold:
                converged = True
                break
        
        # Identify defaults (banks that cannot fully pay)
        default_set = []
        recovery_rates = np.ones(n)
        
        for i in range(n):
            if total_obligations[i] > 0:
                recovery_rates[i] = payment_vector[i] / total_obligations[i]
                recovery_rates[i] = max(self.recovery_rate_floor, recovery_rates[i])
                
                if recovery_rates[i] < 1.0 - 1e-6:
                    default_set.append(i)
        
        # Calculate total losses
        total_losses = np.sum(total_obligations - payment_vector)
        
        return ClearingResult(
            payment_vector=payment_vector,
            default_set=default_set,
            recovery_rates=recovery_rates,
            total_defaults=len(default_set),
            total_losses=total_losses,
            iterations=iteration + 1,
            converged=converged
        )
    
    def compute_contagion_losses(self,
                                 liability_matrix: np.ndarray,
                                 external_assets: np.ndarray,
                                 external_liabilities: np.ndarray,
                                 initial_shock: Dict[int, float]) -> Tuple[ClearingResult, np.ndarray]:
        """
        Compute losses after applying an initial shock to specific banks.
        
        Args:
            liability_matrix: Interbank liability matrix
            external_assets: External assets vector
            external_liabilities: External liabilities vector
            initial_shock: Dict mapping bank_id -> loss amount
            
        Returns:
            Tuple of (ClearingResult, loss_vector per bank)
        """
        # Apply initial shock
        shocked_assets = external_assets.copy()
        for bank_id, loss in initial_shock.items():
            shocked_assets[bank_id] = max(0, shocked_assets[bank_id] - loss)
        
        # Run clearing
        result = self.clear(liability_matrix, shocked_assets, external_liabilities)
        
        # Calculate per-bank losses
        total_obligations = np.sum(liability_matrix, axis=1)
        expected_receipts = liability_matrix.T @ np.ones(len(external_assets))
        actual_receipts = liability_matrix.T @ result.recovery_rates
        
        # Loss = expected receipts - actual receipts + direct shock
        losses = expected_receipts - actual_receipts
        for bank_id, shock in initial_shock.items():
            losses[bank_id] += shock
        
        return result, losses
    
    def stress_test(self,
                    liability_matrix: np.ndarray,
                    external_assets: np.ndarray,
                    external_liabilities: np.ndarray,
                    shock_scenarios: List[Dict[int, float]]) -> List[ClearingResult]:
        """
        Run multiple stress test scenarios.
        
        Args:
            liability_matrix: Interbank liability matrix
            external_assets: External assets vector
            external_liabilities: External liabilities vector
            shock_scenarios: List of shock scenarios
            
        Returns:
            List of ClearingResults for each scenario
        """
        results = []
        for scenario in shock_scenarios:
            result, _ = self.compute_contagion_losses(
                liability_matrix, external_assets, external_liabilities, scenario
            )
            results.append(result)
        return results
    
    def find_critical_banks(self,
                            liability_matrix: np.ndarray,
                            external_assets: np.ndarray,
                            external_liabilities: np.ndarray,
                            shock_fraction: float = 0.5) -> List[Tuple[int, int]]:
        """
        Identify systemically important banks by testing individual failures.
        
        Args:
            liability_matrix: Interbank liability matrix
            external_assets: External assets vector
            external_liabilities: External liabilities vector
            shock_fraction: Fraction of assets lost in shock
            
        Returns:
            List of (bank_id, cascade_size) sorted by impact
        """
        n = len(external_assets)
        impacts = []
        
        for bank_id in range(n):
            # Shock this bank
            shock = {bank_id: external_assets[bank_id] * shock_fraction}
            result, _ = self.compute_contagion_losses(
                liability_matrix, external_assets, external_liabilities, shock
            )
            impacts.append((bank_id, result.total_defaults))
        
        # Sort by cascade size (descending)
        impacts.sort(key=lambda x: x[1], reverse=True)
        return impacts
