"""
Bank Registry: Loads real bank data and provides lookup/selection capabilities.
Also supports synthetic bank generation for stress scenarios.
"""

import os
import json
import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class RealBankConfig:
    """Configuration for a real bank from the registry."""
    bank_id: int
    name: str
    tier: int
    initial_cash: float
    initial_assets: float
    initial_external_liabilities: float
    min_capital_ratio: float
    metadata: Dict[str, Any]

    def to_dict(self) -> dict:
        return {
            "bank_id": self.bank_id,
            "name": self.name,
            "tier": self.tier,
            "initial_cash": self.initial_cash,
            "initial_assets": self.initial_assets,
            "initial_external_liabilities": self.initial_external_liabilities,
            "min_capital_ratio": self.min_capital_ratio,
            "metadata": self.metadata
        }


class BankRegistry:
    """
    Registry of real and synthetic banks.
    Loads real bank data from JSON file extracted from RBI data.
    Provides bank selection, lookup, and synthetic generation.
    """

    def __init__(self):
        self._banks: Dict[int, RealBankConfig] = {}
        self._banks_by_name: Dict[str, RealBankConfig] = {}
        self._loaded = False
        self._data_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "real_banks.json"
        )

    def load(self) -> None:
        """Load bank data from JSON file."""
        if self._loaded:
            return

        if not os.path.exists(self._data_path):
            raise FileNotFoundError(
                f"Bank data file not found: {self._data_path}. "
                "Run extract_pdf.py to generate it."
            )

        with open(self._data_path, 'r') as f:
            data = json.load(f)

        for entry in data.get("banks", []):
            bank = RealBankConfig(
                bank_id=entry["bank_id"],
                name=entry["name"],
                tier=entry["tier"],
                initial_cash=entry["initial_cash"],
                initial_assets=entry["initial_assets"],
                initial_external_liabilities=entry["initial_external_liabilities"],
                min_capital_ratio=entry.get("min_capital_ratio", 0.08),
                metadata=entry.get("metadata", {})
            )
            self._banks[bank.bank_id] = bank
            self._banks_by_name[bank.name] = bank

        self._loaded = True

    def _ensure_loaded(self):
        if not self._loaded:
            self.load()

    def get_all_banks(self) -> List[RealBankConfig]:
        """Get all available real banks."""
        self._ensure_loaded()
        return list(self._banks.values())

    def get_bank_by_id(self, bank_id: int) -> Optional[RealBankConfig]:
        """Get a bank by its registry ID."""
        self._ensure_loaded()
        return self._banks.get(bank_id)

    def get_bank_by_name(self, name: str) -> Optional[RealBankConfig]:
        """Get a bank by name."""
        self._ensure_loaded()
        return self._banks_by_name.get(name)

    def get_banks_by_ids(self, bank_ids: List[int]) -> List[RealBankConfig]:
        """Get multiple banks by their registry IDs."""
        self._ensure_loaded()
        result = []
        for bid in bank_ids:
            bank = self._banks.get(bid)
            if bank:
                result.append(bank)
        return result

    def get_banks_by_names(self, names: List[str]) -> List[RealBankConfig]:
        """Get multiple banks by name."""
        self._ensure_loaded()
        result = []
        for name in names:
            bank = self._banks_by_name.get(name)
            if bank:
                result.append(bank)
        return result

    def get_banks_by_tier(self, tier: int) -> List[RealBankConfig]:
        """Get all banks of a specific tier."""
        self._ensure_loaded()
        return [b for b in self._banks.values() if b.tier == tier]

    def get_available_list(self) -> List[Dict[str, Any]]:
        """Get a summary list of all available banks for the frontend."""
        self._ensure_loaded()
        result = []
        for bank in self._banks.values():
            result.append({
                "id": bank.bank_id,
                "name": bank.name,
                "tier": bank.tier,
                "initial_cash": bank.initial_cash,
                "initial_assets": bank.initial_assets,
                "initial_external_liabilities": bank.initial_external_liabilities,
                "crar": bank.metadata.get("crar", 0),
                "net_npa_ratio": bank.metadata.get("net_npa_ratio", 0),
                "return_on_assets": bank.metadata.get("return_on_assets", 0),
                "return_on_equity": bank.metadata.get("return_on_equity", 0),
                "deposits_crores": bank.metadata.get("deposits_crores", 0),
                "category": "SBI & Associates" if "State Bank" in bank.name else
                           "Nationalised" if bank.tier == 1 and "State Bank" not in bank.name else
                           "Private Sector"
            })
        return result

    def generate_synthetic_banks(
        self,
        count: int,
        start_id: int = 100,
        stress_level: str = "normal",
        seed: Optional[int] = None
    ) -> List[RealBankConfig]:
        """
        Generate synthetic banks for scenarios not achievable with real data.

        Args:
            count: Number of synthetic banks to generate
            start_id: Starting bank_id for synthetic banks
            stress_level: 'normal', 'weak', 'distressed' - controls financial health
            seed: Random seed
        """
        rng = np.random.default_rng(seed)

        # Stress level parameters
        profiles = {
            "normal": {
                "cash_range": (5000, 15000),
                "assets_range": (10000, 30000),
                "ext_liab_range": (3000, 10000),
                "crar_range": (12, 18),
                "npa_range": (0.5, 2.0),
            },
            "weak": {
                "cash_range": (2000, 6000),
                "assets_range": (8000, 20000),
                "ext_liab_range": (5000, 15000),
                "crar_range": (9, 12),
                "npa_range": (3.0, 8.0),
            },
            "distressed": {
                "cash_range": (500, 3000),
                "assets_range": (5000, 15000),
                "ext_liab_range": (8000, 20000),
                "crar_range": (4, 9),
                "npa_range": (8.0, 20.0),
            }
        }

        profile = profiles.get(stress_level, profiles["normal"])
        banks = []

        for i in range(count):
            bid = start_id + i
            tier = 1 if rng.random() < 0.2 else 2
            size_mult = 2.0 if tier == 1 else 1.0

            bank = RealBankConfig(
                bank_id=bid,
                name=f"Synthetic Bank {bid}",
                tier=tier,
                initial_cash=round(rng.uniform(*profile["cash_range"]) * size_mult, 2),
                initial_assets=round(rng.uniform(*profile["assets_range"]) * size_mult, 2),
                initial_external_liabilities=round(rng.uniform(*profile["ext_liab_range"]) * size_mult, 2),
                min_capital_ratio=0.08,
                metadata={
                    "crar": round(rng.uniform(*profile["crar_range"]), 2),
                    "net_npa_ratio": round(rng.uniform(*profile["npa_range"]), 2),
                    "return_on_equity": round(rng.uniform(5, 20), 2),
                    "return_on_assets": round(rng.uniform(0.3, 2.0), 2),
                    "source": "synthetic",
                    "stress_level": stress_level,
                    "data_year": "synthetic"
                }
            )
            banks.append(bank)

        return banks

    def search_banks(self, query: str) -> List[RealBankConfig]:
        """Search banks by name (case-insensitive partial match)."""
        self._ensure_loaded()
        query_lower = query.lower()
        return [b for b in self._banks.values() if query_lower in b.name.lower()]


# Global registry instance
bank_registry = BankRegistry()
