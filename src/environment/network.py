"""
Financial Network: Network topology generation and management.
Uses Barabási–Albert preferential attachment for scale-free networks.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .bank import Bank, BankStatus


@dataclass
class NetworkStats:
    """Network-level statistics."""
    num_banks: int
    num_edges: int
    density: float
    avg_degree: float
    clustering_coefficient: float
    num_defaulted: int
    num_stressed: int
    total_exposure: float
    avg_capital_ratio: float
    
    def to_dict(self) -> dict:
        return {
            'num_banks': self.num_banks,
            'num_edges': self.num_edges,
            'density': self.density,
            'avg_degree': self.avg_degree,
            'clustering_coefficient': self.clustering_coefficient,
            'num_defaulted': self.num_defaulted,
            'num_stressed': self.num_stressed,
            'total_exposure': self.total_exposure,
            'avg_capital_ratio': self.avg_capital_ratio
        }


class FinancialNetwork:
    """
    Financial network representing interbank relationships.
    
    Uses Barabási–Albert model for realistic core-periphery structure.
    """
    
    def __init__(self,
                 num_banks: int = 30,
                 edges_per_node: int = 2,
                 core_fraction: float = 0.2,
                 seed: Optional[int] = None):
        """
        Initialize the financial network.
        
        Args:
            num_banks: Total number of banks (N)
            edges_per_node: Edges per new node in BA model (m)
            core_fraction: Fraction of tier-1 (core) banks
            seed: Random seed for reproducibility
        """
        self.num_banks = num_banks
        self.edges_per_node = edges_per_node
        self.core_fraction = core_fraction
        self.seed = seed
        
        self.graph: nx.DiGraph = None
        self.banks: Dict[int, Bank] = {}
        self.liability_matrix: np.ndarray = None
        
        self._rng = np.random.default_rng(seed)
    
    def generate_network(self,
                         initial_cash_range: Tuple[float, float] = (100, 500),
                         initial_assets_range: Tuple[float, float] = (200, 1000),
                         initial_ext_liab_range: Tuple[float, float] = (50, 300),
                         min_capital_ratio: float = 0.08) -> None:
        """
        Generate the network topology and initialize banks.
        
        Args:
            initial_cash_range: (min, max) for initial cash
            initial_assets_range: (min, max) for initial illiquid assets
            initial_ext_liab_range: (min, max) for external liabilities
            min_capital_ratio: Minimum capital ratio requirement
        """
        # Generate BA scale-free network
        undirected = nx.barabasi_albert_graph(
            n=self.num_banks,
            m=self.edges_per_node,
            seed=self.seed
        )
        
        # Convert to directed graph (both directions for interbank relationships)
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(range(self.num_banks))
        
        for u, v in undirected.edges():
            # Randomly assign direction with some probability of bidirectional
            if self._rng.random() < 0.5:
                self.graph.add_edge(u, v)
            else:
                self.graph.add_edge(v, u)
            
            # Add reverse edge with lower probability (creates debt cycles)
            if self._rng.random() < 0.3:
                self.graph.add_edge(v, u)
        
        # Determine core banks (highest degree centrality)
        degree_centrality = nx.degree_centrality(self.graph)
        sorted_banks = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
        num_core = max(1, int(self.num_banks * self.core_fraction))
        core_bank_ids = {bank_id for bank_id, _ in sorted_banks[:num_core]}
        
        # Initialize banks
        self.banks = {}
        for bank_id in range(self.num_banks):
            tier = 1 if bank_id in core_bank_ids else 2
            
            # Core banks have larger balance sheets
            size_multiplier = 2.0 if tier == 1 else 1.0
            
            initial_cash = self._rng.uniform(*initial_cash_range) * size_multiplier
            initial_assets = self._rng.uniform(*initial_assets_range) * size_multiplier
            initial_ext_liab = self._rng.uniform(*initial_ext_liab_range) * size_multiplier
            
            bank = Bank(
                bank_id=bank_id,
                tier=tier,
                initial_cash=initial_cash,
                initial_assets=initial_assets,
                initial_external_liabilities=initial_ext_liab,
                min_capital_ratio=min_capital_ratio
            )
            self.banks[bank_id] = bank
        
        # Initialize interbank exposures
        self._initialize_interbank_exposures()
        
        # Build liability matrix
        self._build_liability_matrix()
    
    def generate_from_real_banks(self, bank_configs: List[Dict]) -> None:
        """
        Generate network using real bank configurations from the registry.
        
        Args:
            bank_configs: List of dicts with keys:
                bank_id, name, tier, initial_cash, initial_assets,
                initial_external_liabilities, min_capital_ratio, metadata
        """
        self.num_banks = len(bank_configs)
        
        # Generate BA scale-free network topology
        undirected = nx.barabasi_albert_graph(
            n=self.num_banks,
            m=min(self.edges_per_node, self.num_banks - 1),
            seed=self.seed
        )
        
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(range(self.num_banks))
        
        for u, v in undirected.edges():
            if self._rng.random() < 0.5:
                self.graph.add_edge(u, v)
            else:
                self.graph.add_edge(v, u)
            if self._rng.random() < 0.3:
                self.graph.add_edge(v, u)
        
        # Create banks from real configs
        self.banks = {}
        for i, cfg in enumerate(bank_configs):
            bank = Bank(
                bank_id=i,
                tier=cfg.get('tier', 2),
                initial_cash=cfg.get('initial_cash', 500),
                initial_assets=cfg.get('initial_assets', 1000),
                initial_external_liabilities=cfg.get('initial_external_liabilities', 500),
                min_capital_ratio=cfg.get('min_capital_ratio', 0.08)
            )
            # Attach real bank metadata
            bank.name = cfg.get('name', f'Bank {i}')
            bank.metadata = cfg.get('metadata', {})
            self.banks[i] = bank
        
        # Initialize interbank exposures and liability matrix
        self._initialize_interbank_exposures()
        self._build_liability_matrix()
    
    def _initialize_interbank_exposures(self) -> None:
        """Initialize interbank lending relationships based on network edges."""
        for u, v in self.graph.edges():
            # u owes money to v
            lender = self.banks[v]
            borrower = self.banks[u]
            
            # Exposure proportional to lender's size
            max_exposure = lender.balance_sheet.total_assets * 0.1
            exposure = self._rng.uniform(0.01, 0.05) * max_exposure
            
            # Record the exposure
            borrower.add_interbank_liability(v, exposure)
            lender.add_interbank_asset(u, exposure)
            
            # Set edge weight
            self.graph[u][v]['weight'] = exposure
    
    def _build_liability_matrix(self) -> None:
        """Build the liability matrix L where L[i,j] = amount i owes to j."""
        self.liability_matrix = np.zeros((self.num_banks, self.num_banks))
        
        for i in range(self.num_banks):
            bank = self.banks[i]
            for j, amount in bank.balance_sheet.interbank_liabilities.items():
                self.liability_matrix[i, j] = amount
    
    def get_neighbors(self, bank_id: int) -> List[int]:
        """Get neighboring bank IDs (both creditors and debtors)."""
        predecessors = list(self.graph.predecessors(bank_id))
        successors = list(self.graph.successors(bank_id))
        return list(set(predecessors + successors))
    
    def get_creditors(self, bank_id: int) -> List[int]:
        """Get banks that this bank owes money to."""
        return list(self.graph.successors(bank_id))
    
    def get_debtors(self, bank_id: int) -> List[int]:
        """Get banks that owe money to this bank."""
        return list(self.graph.predecessors(bank_id))
    
    def get_exposure(self, from_bank: int, to_bank: int) -> float:
        """Get the exposure (amount owed) from one bank to another."""
        if self.graph.has_edge(from_bank, to_bank):
            return self.graph[from_bank][to_bank].get('weight', 0.0)
        return 0.0
    
    def update_exposure(self, from_bank: int, to_bank: int, amount: float) -> None:
        """Update the exposure between two banks."""
        if amount > 0:
            if not self.graph.has_edge(from_bank, to_bank):
                self.graph.add_edge(from_bank, to_bank)
            self.graph[from_bank][to_bank]['weight'] = amount
            self.liability_matrix[from_bank, to_bank] = amount
        else:
            if self.graph.has_edge(from_bank, to_bank):
                self.graph.remove_edge(from_bank, to_bank)
            self.liability_matrix[from_bank, to_bank] = 0.0
    
    def execute_lending(self, 
                        lender_id: int, 
                        borrower_id: int, 
                        amount: float,
                        interest_rate: float = 0.05) -> bool:
        """
        Execute a lending transaction between two banks.
        
        Args:
            lender_id: Bank providing the loan
            borrower_id: Bank receiving the loan
            amount: Loan amount
            interest_rate: Interest rate on the loan
            
        Returns:
            True if transaction was successful
        """
        lender = self.banks[lender_id]
        borrower = self.banks[borrower_id]
        
        # Check if lender has sufficient excess cash
        if lender.excess_cash < amount:
            return False
        
        # Execute transaction
        lender.make_payment(amount)
        borrower.receive_payment(amount)
        
        # Record obligation (borrower owes lender)
        obligation = amount * (1 + interest_rate)
        borrower.add_interbank_liability(lender_id, obligation)
        lender.add_interbank_asset(borrower_id, obligation)
        
        # Update network
        current_exposure = self.get_exposure(borrower_id, lender_id)
        self.update_exposure(borrower_id, lender_id, current_exposure + obligation)
        
        return True
    
    def execute_repayment(self, 
                          debtor_id: int, 
                          creditor_id: int, 
                          amount: float) -> float:
        """
        Execute a repayment from debtor to creditor.
        
        Args:
            debtor_id: Bank making payment
            creditor_id: Bank receiving payment
            amount: Requested payment amount
            
        Returns:
            Actual amount paid
        """
        debtor = self.banks[debtor_id]
        creditor = self.banks[creditor_id]
        
        # Get current obligation
        current_obligation = self.get_exposure(debtor_id, creditor_id)
        if current_obligation <= 0:
            return 0.0
        
        # Calculate actual payment
        max_payment = min(amount, current_obligation, debtor.balance_sheet.cash)
        
        if max_payment <= 0:
            return 0.0
        
        # Execute payment
        actual_paid = debtor.make_payment(max_payment)
        creditor.receive_payment(actual_paid)
        
        # Update obligations
        debtor.reduce_interbank_liability(creditor_id, actual_paid)
        creditor.reduce_interbank_asset(debtor_id, actual_paid)
        
        # Update network
        new_exposure = current_obligation - actual_paid
        self.update_exposure(debtor_id, creditor_id, new_exposure)
        
        return actual_paid
    
    def get_network_stats(self) -> NetworkStats:
        """Calculate network-level statistics."""
        num_defaulted = sum(1 for b in self.banks.values() if b.status == BankStatus.DEFAULTED)
        num_stressed = sum(1 for b in self.banks.values() if b.status == BankStatus.STRESSED)
        
        total_exposure = np.sum(self.liability_matrix)
        avg_capital_ratio = np.mean([b.capital_ratio for b in self.banks.values()])
        
        return NetworkStats(
            num_banks=self.num_banks,
            num_edges=self.graph.number_of_edges(),
            density=nx.density(self.graph),
            avg_degree=np.mean([d for _, d in self.graph.degree()]),
            clustering_coefficient=nx.average_clustering(self.graph.to_undirected()),
            num_defaulted=num_defaulted,
            num_stressed=num_stressed,
            total_exposure=total_exposure,
            avg_capital_ratio=avg_capital_ratio
        )
    
    def get_centrality_metrics(self) -> Dict[int, Dict[str, float]]:
        """Calculate centrality metrics for each bank."""
        degree_cent = nx.degree_centrality(self.graph)
        betweenness_cent = nx.betweenness_centrality(self.graph)
        
        # PageRank-style importance
        try:
            pagerank = nx.pagerank(self.graph, weight='weight')
        except:
            pagerank = {i: 1.0 / self.num_banks for i in range(self.num_banks)}
        
        metrics = {}
        for bank_id in range(self.num_banks):
            metrics[bank_id] = {
                'degree_centrality': degree_cent.get(bank_id, 0),
                'betweenness_centrality': betweenness_cent.get(bank_id, 0),
                'pagerank': pagerank.get(bank_id, 0)
            }
        
        return metrics
    
    def get_global_state(self) -> np.ndarray:
        """Get global state vector for centralized critic."""
        # Aggregate statistics
        cash_values = [b.balance_sheet.cash for b in self.banks.values()]
        equity_values = [b.balance_sheet.equity for b in self.banks.values()]
        cr_values = [b.capital_ratio for b in self.banks.values()]
        
        stats = self.get_network_stats()
        centrality = self.get_centrality_metrics()
        
        global_state = np.array([
            np.mean(cash_values),
            np.std(cash_values),
            np.mean(equity_values),
            np.std(equity_values),
            np.mean(cr_values),
            np.std(cr_values),
            stats.num_defaulted / self.num_banks,
            stats.num_stressed / self.num_banks,
            stats.total_exposure / (self.num_banks * 1000),  # Normalized
            stats.avg_degree / self.num_banks,
            stats.clustering_coefficient,
            np.mean([c['pagerank'] for c in centrality.values()])
        ], dtype=np.float32)
        
        return global_state
    
    def reset(self,
              initial_cash_range: Tuple[float, float] = (100, 500),
              initial_assets_range: Tuple[float, float] = (200, 1000),
              initial_ext_liab_range: Tuple[float, float] = (50, 300)) -> None:
        """Reset all banks to initial state."""
        for bank_id, bank in self.banks.items():
            size_multiplier = 2.0 if bank.tier == 1 else 1.0
            
            initial_cash = self._rng.uniform(*initial_cash_range) * size_multiplier
            initial_assets = self._rng.uniform(*initial_assets_range) * size_multiplier
            initial_ext_liab = self._rng.uniform(*initial_ext_liab_range) * size_multiplier
            
            bank.reset(initial_cash, initial_assets, initial_ext_liab)
        
        # Reinitialize exposures
        self._initialize_interbank_exposures()
        self._build_liability_matrix()
    
    def to_dict(self) -> dict:
        """Convert network state to dictionary."""
        return {
            'num_banks': self.num_banks,
            'edges': list(self.graph.edges()),
            'banks': {bid: bank.balance_sheet.to_dict() for bid, bank in self.banks.items()},
            'stats': self.get_network_stats().to_dict()
        }
