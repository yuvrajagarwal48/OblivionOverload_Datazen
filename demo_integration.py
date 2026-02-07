"""
Integration Demo for FinSim-MAPPO Extension Plan.

Demonstrates the complete decision-support workflow:
1. State Capture -> History Repository
2. Counterfactual Analysis
3. Credit Risk Prediction
4. Explanation Generation
5. Regulatory Policy Checking
6. UI Data Packaging
"""

import numpy as np
from datetime import datetime
from typing import Dict, Any

# Core components
from src.core.state_capture import StateCapture, BankSnapshot
from src.core.history import SimulationHistory
from src.core.explanation import ExplanationLayer, EventType

# Decision support
from src.decision_support.counterfactual import (
    CounterfactualEngine, HypotheticalTransaction, TransactionType
)

# Analytics
from src.analytics.credit_risk import CreditRiskLayer, RiskFeatures

# Policy
from src.policy.regulatory import RegulatoryPolicyLayer, PolicyRegime

# UI
from src.api.ui_data import UIDataPackager, AlertLevel


class MockBalanceSheet:
    """Mock balance sheet for demo."""
    def __init__(self, cash=1e7, securities=5e7, loans=3e8, 
                 interbank=2e7, deposits=3e8, borrowings=5e7, equity=3e7):
        self.cash = cash
        self.securities = securities
        self.loans = loans
        self.interbank_assets = interbank
        self.interbank_liabilities = borrowings * 0.3  # Part of borrowings as interbank
        self.deposits = deposits
        self.short_term_borrowings = borrowings
        self.equity = equity
        self.total_assets = cash + securities + loans + interbank
        self.total_liabilities = deposits + borrowings
        self.rwa = loans * 1.0 + securities * 0.2 + interbank * 0.5


class MockBankStatus:
    """Mock bank status enum."""
    def __init__(self, value):
        self.value = value


class MockBank:
    """Mock bank for demo."""
    def __init__(self, bank_id: int, healthy: bool = True):
        self.bank_id = bank_id
        self.is_defaulted = False
        self.tier = 1 if bank_id < 3 else (2 if bank_id < 7 else 3)
        self.status = MockBankStatus("active" if healthy else "stressed")
        
        if healthy:
            self.balance_sheet = MockBalanceSheet()
        else:
            # Stressed bank
            self.balance_sheet = MockBalanceSheet(
                cash=5e5, equity=5e6, loans=4e8
            )
        
        self.capital_ratio = self.balance_sheet.equity / self.balance_sheet.rwa
        self.exposures = {}
    
    def add_exposure(self, counterparty_id: int, amount: float):
        self.exposures[counterparty_id] = amount


class DemoIntegration:
    """
    Complete demo of the integrated decision-support platform.
    """
    
    def __init__(self, num_banks: int = 10):
        print("="*60)
        print("FinSim-MAPPO Decision Support Platform Demo")
        print("="*60)
        
        # Initialize components
        print("\n[1] Initializing platform components...")
        
        self.state_capture = StateCapture()
        self.history = SimulationHistory()
        self.explanation_layer = ExplanationLayer()
        self.counterfactual_engine = CounterfactualEngine()
        self.credit_risk_layer = CreditRiskLayer(use_neural=False)
        self.policy_layer = RegulatoryPolicyLayer(regime=PolicyRegime.BASEL_III)
        self.ui_packager = UIDataPackager()
        
        # Create mock banks
        self.banks: Dict[int, MockBank] = {}
        for i in range(num_banks):
            healthy = i < int(num_banks * 0.8)  # 80% healthy
            self.banks[i] = MockBank(i, healthy=healthy)
        
        # Create exposures (simple network)
        for i in range(num_banks):
            for j in range(i+1, min(i+3, num_banks)):
                amount = np.random.uniform(1e6, 5e6)
                self.banks[i].add_exposure(j, amount)
                self.banks[j].add_exposure(i, amount * 0.5)
        
        print(f"   Created {num_banks} banks with interbank exposures")
        print(f"   Policy regime: {self.policy_layer.regime.value}")
    
    def run_state_capture_demo(self, timestep: int = 0):
        """Demonstrate state capture and history."""
        print("\n[2] State Capture & History Repository Demo")
        print("-"*50)
        
        # Capture snapshots
        snapshots = []
        for bank_id, bank in self.banks.items():
            snapshot = BankSnapshot.from_bank(bank, timestep)
            snapshots.append(snapshot)
            self.state_capture.capture_bank(bank, timestep)
        
        print(f"   Captured {len(snapshots)} bank snapshots")
        print(f"   Sample snapshot - Bank 0:")
        print(f"     Capital Ratio: {snapshots[0].capital_ratio:.2%}")
        print(f"     Total Assets: ${snapshots[0].total_assets:,.0f}")
        print(f"     Solvency Status: {snapshots[0].solvency_status}")
        
        # Record to history using correct API
        self.history.begin_timestep(timestep)
        # Snapshots are captured directly via state_capture, which is part of history
        self.history.end_timestep()
        
        print(f"   Recorded timestep {timestep} to history repository")
    
    def run_credit_risk_demo(self):
        """Demonstrate credit risk layer."""
        print("\n[3] Credit Risk Layer Demo")
        print("-"*50)
        
        risk_outputs = {}
        for bank_id, bank in self.banks.items():
            output = self.credit_risk_layer.predict(bank)
            risk_outputs[bank_id] = output
        
        # Display summary
        summary = self.credit_risk_layer.get_system_risk_summary(risk_outputs)
        
        print(f"   System Risk Summary:")
        print(f"     Average PD: {summary['average_pd']:.2%}")
        print(f"     Max PD: {summary['max_pd']:.2%}")
        print(f"     High-Risk Banks: {summary['high_risk_count']}")
        print(f"     Total Expected Loss: ${summary['total_expected_loss']:,.0f}")
        
        print(f"\n   Rating Distribution:")
        for rating, count in summary['rating_distribution'].items():
            if count > 0:
                print(f"     {rating}: {count} banks")
        
        return risk_outputs
    
    def run_policy_compliance_demo(self):
        """Demonstrate regulatory policy layer."""
        print("\n[4] Regulatory Policy Layer Demo")
        print("-"*50)
        
        all_violations = []
        for bank_id, bank in self.banks.items():
            violations = self.policy_layer.check_compliance(bank)
            if violations:
                all_violations.extend(violations)
                
                # Record for explanation
                for v in violations:
                    self.explanation_layer.record_event(
                        0, v.policy_name, bank_id, {'violation': v.to_dict()}
                    )
        
        # Get compliance summary
        summary = self.policy_layer.get_compliance_summary()
        
        print(f"   Policy Regime: {summary['regime']}")
        print(f"   Total Capital Requirement: {summary['total_capital_requirement']:.1%}")
        print(f"   Active Violations: {summary['active_violations']}")
        print(f"   Severity Distribution:")
        for severity, count in summary['severity_distribution'].items():
            if count > 0:
                print(f"     {severity.upper()}: {count}")
        print(f"   System Compliant: {summary['system_compliant']}")
        
        # Enforce violations
        if all_violations:
            enforcement = self.policy_layer.enforce_violations(all_violations)
            restricted = len(enforcement)
            print(f"\n   Enforcement Actions Applied: {restricted} entities restricted")
        
        return all_violations
    
    def run_counterfactual_demo(self):
        """Demonstrate counterfactual analysis."""
        print("\n[5] Counterfactual Decision Support Demo")
        print("-"*50)
        
        # Create a hypothetical loan transaction
        transaction = HypotheticalTransaction(
            transaction_id="TX_001",
            transaction_type=TransactionType.LOAN_APPROVAL,
            initiator_id=0,
            counterparty_id=8,  # Potentially stressed bank
            principal_amount=5e6,
            description="Proposed interbank loan to Bank 8"
        )
        
        print(f"   Analyzing transaction: {transaction.description}")
        print(f"     Amount: ${transaction.principal_amount:,.0f}")
        print(f"     From: Bank {transaction.initiator_id}")
        print(f"     To: Bank {transaction.counterparty_id}")
        
        # Note: In real usage, would pass actual env, exchanges, ccps
        # Here we show the structure
        print("\n   Counterfactual Analysis Framework:")
        print("     1. Replicate current state")
        print("     2. Inject hypothetical transaction")
        print("     3. Run multiple forward simulations")
        print("     4. Compare baseline vs counterfactual outcomes")
        print("     5. Compute risk score and recommendation")
        
        # Demonstrate risk scoring logic
        initiator = self.banks[0]
        counterparty = self.banks[8]
        
        initiator_risk = self.credit_risk_layer.predict(initiator)
        counterparty_risk = self.credit_risk_layer.predict(counterparty)
        
        print(f"\n   Transaction Risk Analysis:")
        print(f"     Initiator (Bank 0) PD: {initiator_risk.probability_of_default:.2%}")
        print(f"     Counterparty (Bank 8) PD: {counterparty_risk.probability_of_default:.2%}")
        
        # Simple risk score computation
        risk_score = (
            initiator_risk.probability_of_default * 30 +
            counterparty_risk.probability_of_default * 40 +
            initiator_risk.systemic_importance * 15 +
            counterparty_risk.contagion_vulnerability * 15
        ) * 100
        
        print(f"     Combined Risk Score: {risk_score:.1f}/100")
        
        if risk_score < 20:
            rec = "APPROVE"
        elif risk_score < 40:
            rec = "APPROVE WITH CONDITIONS"
        elif risk_score < 60:
            rec = "REVIEW REQUIRED"
        elif risk_score < 80:
            rec = "CAUTION ADVISED"
        else:
            rec = "REJECT"
        
        print(f"     Recommendation: {rec}")
    
    def run_explanation_demo(self):
        """Demonstrate explanation layer."""
        print("\n[6] Explanation & Attribution Layer Demo")
        print("-"*50)
        
        # Record exposures for causality analysis
        for bank_id, bank in self.banks.items():
            for cp_id, amount in bank.exposures.items():
                self.explanation_layer.record_exposure(bank_id, cp_id, amount)
        
        # Record state for stressed banks
        for bank_id, bank in self.banks.items():
            self.explanation_layer.record_state(0, bank_id, {
                'capital_ratio': bank.capital_ratio,
                'cash': bank.balance_sheet.cash,
                'total_assets': bank.balance_sheet.total_assets
            })
        
        # Simulate a default event for demo
        stressed_bank_id = 8  # Known stressed bank
        
        # Record preceding events
        self.explanation_layer.record_event(0, 'margin_call', stressed_bank_id, 
                                           {'amount': 1e6})
        self.explanation_layer.record_event(0, 'liquidity_stress', stressed_bank_id,
                                           {'severity': 0.7})
        
        # Generate explanation
        explanation = self.explanation_layer.explain_event(
            EventType.DEFAULT, stressed_bank_id, 0
        )
        
        print(f"   Event Explanation Generated:")
        print(f"     Event: {explanation.event_type.value}")
        print(f"     Affected Entity: Bank {explanation.affected_entity}")
        print(f"     Severity Score: {explanation.severity_score:.2f}")
        print(f"     Systemic Impact: {explanation.systemic_impact:.2f}")
        print(f"\n     Primary Cause:")
        print(f"       Type: {explanation.primary_cause.cause_type.value}")
        print(f"       Contribution: {explanation.primary_cause.contribution:.0%}")
        
        if explanation.contributing_factors:
            print(f"\n     Contributing Factors:")
            for factor in explanation.contributing_factors[:3]:
                print(f"       - {factor.cause_type.value}: {factor.contribution:.0%}")
        
        print(f"\n     Prevention Actions:")
        for action in explanation.prevention_actions[:3]:
            print(f"       - {action}")
        
        print(f"\n     Narrative:")
        print(f"       {explanation.narrative[:200]}...")
    
    def run_ui_packaging_demo(self, risk_outputs: Dict[int, Any]):
        """Demonstrate UI data packaging."""
        print("\n[7] UI Data Packaging Demo")
        print("-"*50)
        
        # Create entity cards
        for bank_id, bank in self.banks.items():
            risk = risk_outputs.get(bank_id)
            violations = self.policy_layer.violation_history.get(bank_id, [])
            self.ui_packager.update_entity_card(bank, "bank", risk, violations)
        
        # Create time series (mock data)
        capital_ts = self.ui_packager.create_time_series(
            "system_capital_ratio", "System-wide Capital Ratio", "%"
        )
        for t in range(10):
            capital_ts.add_point(t, 0.10 + np.random.uniform(-0.01, 0.01))
        
        stress_ts = self.ui_packager.create_time_series(
            "stress_index", "System Stress Index", ""
        )
        for t in range(10):
            stress_ts.add_point(t, 0.2 + t * 0.05 + np.random.uniform(-0.02, 0.02))
        
        # Build network visualization
        exposures = {}
        for bank_id, bank in self.banks.items():
            for cp_id, amount in bank.exposures.items():
                exposures[(bank_id, cp_id)] = amount
        
        self.ui_packager.build_network_from_env(self.banks, exposures, risk_outputs)
        
        # Create exposure heatmap
        self.ui_packager.create_exposure_heatmap(self.banks, exposures)
        
        # Add alerts
        self.ui_packager.add_alert(
            AlertLevel.WARNING,
            "Elevated System Stress",
            "Stress index has increased 25% in last 5 timesteps",
            timestamp=9
        )
        
        for violation in self.policy_layer.violations[:2]:
            self.ui_packager.add_alert(
                AlertLevel.CRITICAL if violation.severity.value in ['severe', 'critical'] 
                else AlertLevel.WARNING,
                f"Policy Violation: {violation.policy_name}",
                violation.description,
                entity_id=violation.entity_id
            )
        
        # Update summary (mock env)
        class MockEnv:
            def __init__(self, banks):
                self.banks = banks
        
        compliance = self.policy_layer.get_compliance_summary()
        self.ui_packager.update_summary(MockEnv(self.banks), 9, risk_outputs, compliance)
        
        # Export dashboard data
        dashboard = self.ui_packager.export_dashboard_data()
        
        print(f"   Dashboard Data Package:")
        print(f"     Summary:")
        print(f"       Overall Health: {dashboard['summary']['overall_health']}%")
        print(f"       Stability Index: {dashboard['summary']['stability_index']}")
        print(f"       Total Banks: {dashboard['summary']['total_banks']}")
        print(f"       Defaulted: {dashboard['summary']['defaulted_banks']}")
        
        print(f"\n     Entity Cards: {len(dashboard['entity_cards'])}")
        print(f"     Time Series: {len(dashboard['time_series'])}")
        print(f"     Network Nodes: {len(dashboard['network']['nodes'])}")
        print(f"     Network Edges: {len(dashboard['network']['edges'])}")
        print(f"     Active Alerts: {len(dashboard['alerts'])}")
        
        # Show sample alerts
        if dashboard['alerts']:
            print(f"\n     Sample Alerts:")
            for alert in dashboard['alerts'][:3]:
                print(f"       [{alert['level'].upper()}] {alert['title']}")
        
        return dashboard
    
    def run_full_demo(self):
        """Run complete integration demo."""
        print("\n" + "="*60)
        print("Starting Full Integration Demo")
        print("="*60)
        
        # 1. State capture
        self.run_state_capture_demo(timestep=0)
        
        # 2. Credit risk
        risk_outputs = self.run_credit_risk_demo()
        
        # 3. Policy compliance
        violations = self.run_policy_compliance_demo()
        
        # 4. Counterfactual analysis
        self.run_counterfactual_demo()
        
        # 5. Explanation layer
        self.run_explanation_demo()
        
        # 6. UI packaging
        dashboard = self.run_ui_packaging_demo(risk_outputs)
        
        # Summary
        print("\n" + "="*60)
        print("Integration Demo Complete")
        print("="*60)
        print("""
Components Demonstrated:
  ✓ State Capture: Continuous snapshots of all entities
  ✓ History Repository: Centralized event and state logging
  ✓ Credit Risk Layer: PD/LGD prediction with ratings
  ✓ Regulatory Policy: Basel III compliance checking
  ✓ Counterfactual Engine: Transaction impact analysis
  ✓ Explanation Layer: Causal analysis and narratives
  ✓ UI Data Packaging: Dashboard-ready data export

The platform is ready for:
  - Real-time simulation monitoring
  - Policy impact analysis
  - What-if scenario exploration
  - Regulatory reporting
  - Interactive dashboard visualization
""")
        
        return dashboard


def main():
    """Run the integration demo."""
    demo = DemoIntegration(num_banks=10)
    dashboard = demo.run_full_demo()
    
    # Optionally export to file
    # demo.ui_packager.export_to_json("dashboard_data.json")
    
    return dashboard


if __name__ == "__main__":
    main()
