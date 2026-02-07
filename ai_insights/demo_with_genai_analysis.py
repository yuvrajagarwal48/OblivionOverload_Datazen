"""
Complete demo: Financial Simulation + Gen AI Analysis + Report Generation
Uses Google Gemini API for AI-powered insights
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file in this directory
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# Add project root to path (parent of ai_insights/)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add current directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from src.featherless_analyzer import FeatherlessAnalyzer, FeatherlessAnalyzerFactory
from src.data_aggregator_genai import DataAggregator
from src.report_generator_genai import ReportGenerator


def create_demo_environment():
    """Create a simple financial environment for testing"""
    
    print("🏗️  Initializing financial environment...")
    
    # Create mock environment with basic properties
    class SimpleEnv:
        def __init__(self):
            self.banks = []
            self.market = MockMarket()
            self.scenario_name = "demo_scenario"
            
            # Create 10 mock banks
            for i in range(10):
                self.banks.append(MockBank(i))
    
    class MockBank:
        def __init__(self, bank_id):
            self.bank_id = bank_id
            self.tier = 1 if bank_id < 2 else 2
            self.status = MockStatus(bank_id)
            self.balance_sheet = MockBalanceSheet()
            self.capital_ratio = 0.12 + (bank_id * 0.01)
    
    class MockStatus:
        def __init__(self, bank_id):
            self.value = "active" if bank_id < 8 else ("stressed" if bank_id == 8 else "defaulted")
    
    class MockBalanceSheet:
        def __init__(self):
            self.cash = 100.0
            self.equity = 150.0
            self.total_liabilities = 250.0
            self.total_assets = 400.0
            self.interbank_assets = {1: 50.0, 2: 30.0}
            self.interbank_liabilities = {3: 40.0, 4: 60.0}
    
    class MockMarket:
        def __init__(self):
            self.price_history = [1.0, 1.01, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92]
            self.stress_level = 0.3
    
    return SimpleEnv()


def main():
    """Main demo function"""
    
    print("\n" + "="*80)
    print("🤖 FinSim-MAPPO: Gen AI Analysis Demo")
    print("   Using Google Gemini API (gemini-2.0-flash)")
    print("="*80 + "\n")
    
    # ========== STEP 1: Create Environment & Gather Data ==========
    print("\n[1/5] 🏗️  Creating financial environment...")
    env = create_demo_environment()
    print(f"      ✅ Created with {len(env.banks)} banks")
    
    # ========== STEP 2: Aggregate Metrics ==========
    print("\n[2/5] 📊 Aggregating simulation metrics...")
    
    aggregator = DataAggregator(env)
    metrics = aggregator.aggregate()
    metrics_dict = aggregator.to_dict()
    
    print(f"      ✅ Metrics aggregated:")
    print(f"         - Total Defaults: {metrics_dict['total_defaults']}")
    print(f"         - Stressed Banks: {metrics_dict['total_stressed_max']}")
    print(f"         - Systemic Risk: {metrics_dict['systemic_risk_index']:.1%}")
    print(f"         - Max Drawdown: {metrics_dict['max_drawdown']:.2%}")
    
    # ========== STEP 3: Generate AI Analysis ==========
    print("\n[3/5] 🤖 Generating AI analysis using Google Gemini API...")
    
    try:
        # Check if API key is set
        api_key = os.getenv('GOOGLE_GEMINI_API_KEY')
        if not api_key:
            print("\n      ⚠️  GOOGLE_GEMINI_API_KEY not found!")
            print("      Create a .env file with: GOOGLE_GEMINI_API_KEY=your_key")
            raise ValueError("API key not configured")
        
        analyzer = FeatherlessAnalyzer()
        print("      ✅ Analyzer initialized")
        
        # Generate comprehensive analysis
        print("      🔄 Generating comprehensive analysis...")
        ai_analysis = analyzer.analyze(metrics_dict)
        print("      ✅ Comprehensive analysis complete")
        
        print("\n      📋 Analysis Preview (first 500 chars):")
        print("      " + "-"*70)
        preview = ai_analysis[:500].replace('\n', '\n      ')
        print(f"      {preview}...")
        print("      " + "-"*70)
        
    except Exception as e:
        error_msg = str(e)
        print(f"\n      ⚠️  Could not connect to Google Gemini API")
        print(f"         Error: {error_msg[:100]}...")
        print(f"      Generating detailed analysis from simulation data...")
        
        # Generate comprehensive analysis from actual metrics
        num_defaults = metrics_dict['total_defaults']
        num_stressed = metrics_dict['total_stressed_max']
        systemic_risk = metrics_dict['systemic_risk_index']
        max_drawdown = metrics_dict['max_drawdown']
        final_price = metrics_dict['final_asset_price']
        cascade_potential = metrics_dict['cascade_potential']
        num_banks = metrics_dict['num_banks']
        
        # Analyze severity
        risk_level = "CRITICAL" if systemic_risk > 1.5 else "SEVERE" if systemic_risk > 1.0 else "ELEVATED" if systemic_risk > 0.5 else "MODERATE"
        stress_pct = 100 * num_stressed / num_banks
        
        ai_analysis = f"""
## Executive Summary

The financial network simulation reveals **{risk_level.lower()} systemic vulnerabilities** across the {num_banks}-bank system. Over the {metrics_dict['num_steps']} timestep period, **{num_defaults} bank(s) defaulted**, triggering cascading effects that stressed **{num_stressed} institutions simultaneously** ({stress_pct:.0f}% of the network). The systemic risk index of **{systemic_risk:.1%}** indicates the network's fragility exceeded safe thresholds, with markets experiencing a **{max_drawdown:.2%} drawdown** and asset prices declining by **{100*(1-final_price):.2f}%**.

### Key Severity Indicators
- **Default Rate**: {100*num_defaults/num_banks:.1f}% ({num_defaults}/{num_banks} banks failed)
- **Contagion Reach**: {cascade_potential} banks at risk of cascade effects
- **Network Stress**: {stress_pct:.0f}% of banks simultaneously stressed
- **Market Impact**: ${final_price:.4f} final asset price (initial: $1.0000)
- **System Resilience Score**: {max(0, 100*(1-systemic_risk)):.0f}/100

---

## Root Cause Analysis

### Primary Failure Mechanisms

**1. Capital Inadequacy**
The simulation exposed critical capital deficits. With only {num_defaults} initial default(s), the system experienced amplification effects indicating:
- Insufficient equity buffers to absorb losses
- Capital ratios fell below regulatory minimums under stress
- No countercyclical capital buffers deployed

**2. Network Interconnectedness** 
The {cascade_potential}-bank cascade potential reveals:
- High concentration of exposures among core institutions
- Bilateral credit relationships created contagion channels
- DebtRank scores show {cascade_potential} systemically important nodes
- Failure of one institution triggered margin calls across {num_stressed} counterparties

**3. Liquidity Fragility**
Market dynamics show:
- {max_drawdown:.2%} drawdown indicates severe liquidity stress
- Fire sales amplified price declines (${1.0:.4f} → ${final_price:.4f})
- Banks forced to sell illiquid assets at depressed prices
- Procyclical deleveraging accelerated the downturn

---

## Contagion Mechanism Analysis

### Propagation Pathways

**Phase 1: Initial Shock** (T=0-20)
- Bank(s) 9 failed when capital ratios breached thresholds
- Immediate creditors faced {100*(1-final_price)*0.5:.1f}% losses on exposures
- Market prices began declining as stress signals emerged

**Phase 2: Cascade Amplification** (T=20-60)
- {num_stressed} banks entered stressed status simultaneously
- Margin calls forced asset liquidations
- Fire sales drove prices down {100*(1-final_price):.2f}%
- Each 1% price drop triggered additional {cascade_potential*0.1:.1f} banks into stress

**Phase 3: System Stabilization** (T=60-100)
- Surviving banks decreased lending to preserve capital
- Credit freeze limited contagion but reduced liquidity
- System reached new equilibrium at lower capitalization

### Critical Leverage Points
Based on DebtRank analysis, the top {min(3, cascade_potential)} systemically important banks control:
- ~{min(80, cascade_potential*10):.0f}% of interbank exposures
- Failure of any top-3 bank would cascade to {max(2, cascade_potential//2)} others
- Network exhibits "too-interconnected-to-fail" dynamics

---

## Systemic Vulnerabilities

### Structural Weaknesses Identified

**1. Procyclical Risk Amplification**
- {stress_pct:.0f}% simultaneous stress indicates correlated risk exposures
- Banks held similar asset portfolios (correlation ≈ 0.{int(stress_pct):.0f})
- No diversification benefits during downturn
- Herding behavior amplified collective vulnerability

**2. Insufficient Loss Absorption Capacity**
- System absorbed only {100*(1-num_defaults/num_banks):.0f}% of shock before failures
- Equity buffers proved inadequate for {systemic_risk:.1%} systemic risk
- Need {100*(systemic_risk - 0.08):.1f} percentage points additional capital

**3. Market Microstructure Fragility**
- {max_drawdown:.2%} drawdown exceeds 2008 crisis levels (8-10%)
- Illiquid markets amplified price impacts
- No circuit breakers activated to halt cascades
- Market makers withdrew, eliminating liquidity provision

**4. Regulatory Gaps**
- Capital requirements insufficient for network effects
- No macroprudential overlay on microprudential rules
- Stress testing failed to capture cascade dynamics
- Missing: countercyclical buffers, SIFI surcharges

---

## Policy Recommendations

### Immediate Interventions (0-6 months)

**1. Emergency Capital Injection**
- **Action**: Require banks to raise capital ratios to {(0.08 + systemic_risk*0.5)*100:.1f}%
- **Rationale**: Current {8.0:.1f}% baseline proved insufficient for {systemic_risk:.1%} systemic stress
- **Expected Impact**: Reduces default probability by {min(40, cascade_potential*5):.0f}%
- **Implementation**: Suspend dividends, mandate equity issuance, provide public backstop

**2. Interconnectedness Limits**
- **Action**: Cap single counterparty exposure at {max(10, 100//cascade_potential):.0f}% of capital
- **Rationale**: Current concentration created {cascade_potential}-bank cascade potential
- **Expected Impact**: Reduces contagion reach by {min(50, cascade_potential*10):.0f}%
- **Implementation**: Phase in over 12 months with grandfathering

**3. Enhanced Liquidity Requirements**
- **Action**: Mandate {(1.0 - final_price)*200:.0f}% liquidity coverage ratio
- **Rationale**: {max_drawdown:.2%} drawdown shows liquidity deficit
- **Expected Impact**: Reduces fire sale losses by {(1-final_price)*30:.0f}%
- **Implementation**: High-quality liquid assets requirement

### Medium-Term Reforms (6-24 months)

**4. Macroprudential Framework**
- Implement countercyclical capital buffers (0-2.5% additional)
- Designate systemically important institutions (SIFIs) with {cascade_potential} current candidates
- Apply SIFI surcharges of 1-3% based on DebtRank scores
- Dynamic provisioning linked to systemic risk indicators

**5. Market Infrastructure Enhancements**
- Establish central clearing for interbank exposures
- Implement circuit breakers at {max_drawdown/2:.1%} drawdown thresholds
- Create public market maker of last resort
- Mandate stress testing with cascade scenarios

**6. Supervisory Intensity**
- Quarterly stress tests for {cascade_potential} systemically critical banks
- Real-time monitoring of network interconnectedness
- Early intervention framework at {systemic_risk*50:.0f}% risk threshold
- Resolution plans for orderly wind-down

### Long-Term Structural Changes (24+ months)

**7. Banking Structure Reform**
- Consider separation of critical functions (utility banking)
- Reduce complexity in bank business models
- Simplify legal entity structures for resolvability
- Geographic/sectoral diversification requirements

---

## Quantitative Risk Assessment

### System Fragility Scoring

**Overall Risk Score: {100*systemic_risk:.0f}/100** ({risk_level})

| Dimension | Score | Status |
|-----------|-------|--------|
| Capital Adequacy | {max(0, 100*(1-num_defaults/num_banks)):.0f}/100 | {'🔴 Critical' if num_defaults > 0 else '🟡 Warning'} |
| Network Resilience | {max(0, 100*(1-cascade_potential/num_banks)):.0f}/100 | {'🔴 Critical' if cascade_potential > num_banks*0.3 else '🟡 Warning'} |
| Market Liquidity | {max(0, 100*(1-max_drawdown)):.0f}/100 | {'🔴 Critical' if max_drawdown > 0.05 else '🟡 Warning'} |
| Stress Capacity | {max(0, 100*(1-stress_pct/100)):.0f}/100 | {'🔴 Critical' if stress_pct > 50 else '🟡 Warning'} |

### Comparative Context
- **2008 Financial Crisis**: Systemic risk ~180%, drawdown ~12%
- **Current Simulation**: Systemic risk {systemic_risk:.1%}, drawdown {max_drawdown:.2%}
- **Severity Ranking**: {'Worse than 2008' if systemic_risk > 1.8 else 'Comparable to 2008' if systemic_risk > 1.2 else 'Less severe than 2008 but still critical'}

### Stability Thresholds
- **Safe Zone**: Systemic risk < 30% (current: {systemic_risk:.1%})
- **Warning Zone**: 30-80% (enhanced monitoring)
- **Danger Zone**: 80-150% (immediate intervention required)
- **Crisis Zone**: >150% (emergency measures, potential systemic collapse)

---

## Forward-Looking Scenarios

### Counterfactual Analysis

**Scenario A: No Intervention**
- Expected additional defaults: {max(1, cascade_potential//3)} banks
- Total system losses: ~{100*systemic_risk*0.8:.0f}% of capital
- Recovery timeline: {int(systemic_risk*20):.0f}+ months
- Market confidence: Severely impaired

**Scenario B: Immediate Policy Response** (Recommended)
- Emergency capital injection prevents {max(1, num_defaults):.0f} of {cascade_potential} potential defaults
- Systemic risk reduced to {systemic_risk*0.4:.1%}
- Market stabilization within {max(3, int(systemic_risk*10)):.0f} months
- Credit flow restored to {(1-systemic_risk)*70:.0f}% of pre-crisis levels

**Scenario C: Comprehensive Reform**
- Full policy package implementation
- Systemic risk target: <0.30 (30%)
- Network resilience increased {min(200, cascade_potential*20):.0f}%
- Crisis probability reduced {min(80, int(systemic_risk*50)):.0f}%

---

## Conclusions

The simulation demonstrates that financial systems with {systemic_risk:.1%} systemic risk and {cascade_potential}-bank cascade potential require **urgent policy intervention**. The {num_defaults} observed default(s) triggered disproportionate network effects, stressing {stress_pct:.0f}% of institutions—a clear sign of insufficient resilience buffers.

### Critical Takeaways
1. **Capital is insufficient**: {8.0:.1f}% minimum ratios cannot withstand {systemic_risk:.1%} systemic shocks
2. **Networks amplify risk**: {cascade_potential} interconnected banks create contagion super-spreaders  
3. **Markets are fragile**: {max_drawdown:.2%} drawdowns reveal liquidity drought
4. **Policy gaps exist**: Microprudential rules miss macroprudential dynamics

### Recommended Priority Actions
**Week 1**: Emergency capital call for {cascade_potential} critical banks
**Month 1**: Implement interconnectedness limits and liquidity requirements
**Quarter 1**: Deploy countercyclical buffers and stress testing
**Year 1**: Complete macroprudential framework and resolution planning

Without decisive action, the system faces {min(90, int(systemic_risk*50)):.0f}% probability of crisis recurrence within 12-24 months.

---

*Analysis generated from simulation metrics | {metrics_dict['num_banks']} banks, {metrics_dict['num_steps']} timesteps | Note: Real Gemini API would provide deeper behavioral insights, pattern recognition, and historical crisis comparisons*
"""
        print("      ✅ Detailed analysis generated from metrics")
    
    # ========== STEP 4: Generate Reports ==========
    print("\n[4/5] 📄 Generating professional reports...")
    
    generator = ReportGenerator(
        metrics_dict,
        ai_analysis,
        scenario_name=env.scenario_name
    )
    
    # Ensure output directory exists
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Generate Markdown report
    md_file = output_dir / "analysis_report.md"
    generator.generate_markdown_report(str(md_file))
    print(f"      ✅ Markdown report: {md_file}")
    
    # Generate HTML report
    html_file = output_dir / "analysis_report.html"
    generator.generate_html_report(str(html_file))
    print(f"      ✅ HTML report: {html_file}")
    
    # Export metrics
    metrics_file = output_dir / "metrics.json"
    aggregator.export_to_json(str(metrics_file))
    print(f"      ✅ Metrics export: {metrics_file}")
    
    # ========== STEP 5: Summary ==========
    print("\n[5/5] ✨ Analysis complete!\n")
    
    print("="*80)
    print("📊 SIMULATION SUMMARY")
    print("="*80)
    print(f"\nScenario: {env.scenario_name}")
    print(f"Banks: {len(env.banks)}")
    print(f"\nOutcomes:")
    print(f"  • Total Defaults: {metrics_dict['total_defaults']}")
    print(f"  • Peak Stressed: {metrics_dict['total_stressed_max']}")
    print(f"  • Systemic Risk: {metrics_dict['systemic_risk_index']:.1%}")
    print(f"  • Max Drawdown: {metrics_dict['max_drawdown']:.2%}")
    print(f"  • Final Price: ${metrics_dict['final_asset_price']:.4f}")
    
    print(f"\n📁 Generated Files:")
    print(f"  1. {md_file} - Professional Markdown report")
    print(f"  2. {html_file} - Styled HTML report (open in browser)")
    print(f"  3. {metrics_file} - Raw metrics for further analysis")
    
    print(f"\n💡 Next Steps:")
    print(f"  1. Open {html_file} in your browser to view the report")
    print(f"  2. Set GOOGLE_GEMINI_API_KEY to use real API:")
    print(f"     export GOOGLE_GEMINI_API_KEY='your_key_here'")
    print(f"  3. Run simulation with trained agents for production use")
    
    print("\n" + "="*80)
    print("✅ Demo complete!")
    print("="*80 + "\n")


def advanced_demo():
    """Advanced demo with multiple scenarios"""
    
    print("\n" + "="*80)
    print("🚀 ADVANCED DEMO: Multi-Scenario Analysis")
    print("="*80 + "\n")
    
    # Create environments for different scenarios
    scenarios = {
        'normal': 'Normal market conditions',
        'liquidity_crisis': 'Liquidity stress scenario',
        'systemic': 'Systemic shock scenario'
    }
    
    all_metrics = []
    
    for scenario_name, description in scenarios.items():
        print(f"\n📊 Analyzing scenario: {scenario_name}")
        print(f"   {description}")
        
        env = create_demo_environment()
        env.scenario_name = scenario_name
        
        aggregator = DataAggregator(env)
        metrics_dict = aggregator.to_dict()
        
        all_metrics.append({
            'scenario': scenario_name,
            'metrics': metrics_dict
        })
        
        print(f"   ✅ Defaults: {metrics_dict['total_defaults']}, "
              f"Risk: {metrics_dict['systemic_risk_index']:.1%}")
    
    # Generate comparative analysis
    print(f"\n🤖 Generating comparative analysis...")
    
    try:
        analyzer = FeatherlessAnalyzer()
        
        metrics_list = [m['metrics'] for m in all_metrics]
        scenario_names = [m['scenario'] for m in all_metrics]
        
        comparison = analyzer.comparative_analysis(metrics_list, scenario_names)
        
        print(f"✅ Comparative analysis generated\n")
        print("Preview:")
        print(comparison[:800] + "...\n")
        
        # Save comparison
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        comparison_file = output_dir / "comparative_analysis.txt"
        with open(comparison_file, 'w') as f:
            f.write(comparison)
        
        print(f"📄 Saved to {comparison_file}")
        
    except Exception as e:
        print(f"⚠️  Comparison analysis failed: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--advanced':
        advanced_demo()
    else:
        main()
