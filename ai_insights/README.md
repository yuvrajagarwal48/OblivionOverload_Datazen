# 🤖 AI Insights & Analysis Module

This module provides **Gen AI-powered analysis** for financial system simulation results using Google Gemini API.

## 📁 Structure

```
ai_insights/
├── src/                          # Core analysis modules
│   ├── featherless_analyzer.py   # Gemini API integration
│   ├── data_aggregator_genai.py  # Metrics extraction & aggregation
│   └── report_generator_genai.py # Professional report generation
├── demo_with_genai_analysis.py   # Demo script with mock data
├── outputs/                      # Generated reports
│   ├── analysis_report.html      # Styled HTML report
│   ├── analysis_report.md        # Markdown report
│   └── metrics.json              # Raw metrics data
├── .env                          # API keys (not committed)
├── .env.example                  # Template for API keys
└── README.md                     # This file
```

## 🚀 Features

### 1. **Comprehensive Metrics Aggregation** (`data_aggregator_genai.py`)
- Capital ratio tracking
- Default & stress event detection
- DebtRank computation
- Systemic risk index calculation
- Network contagion analysis

### 2. **Gen AI Analysis** (`featherless_analyzer.py`)
- Uses Google Gemini API (gemini-2.0-flash)
- Deep causal analysis
- Policy recommendations
- Risk scenario evaluation
- Automatic fallback with structured analysis

### 3. **Professional Report Generation** (`report_generator_genai.py`)
- **Markdown Reports**: Version-control friendly
- **HTML Reports**: Styled, interactive dashboards
- **JSON Exports**: Raw data for further processing

## ⚙️ Setup

### 1. Install Dependencies

```bash
pip install google-generativeai>=0.3.0 python-dotenv
```

### 2. Configure API Key

```bash
# Copy template
cp .env.example .env

# Add your Gemini API key
echo "GOOGLE_GEMINI_API_KEY=your_key_here" > .env
```

### 3. Run Demo

```bash
python demo_with_genai_analysis.py
```

## 📊 Usage with Real Simulation

```python
from src.environment.financial_env import FinancialEnvironment
from ai_insights.src.data_aggregator_genai import DataAggregatorGenAI
from ai_insights.src.featherless_analyzer import FeatherlessAnalyzer
from ai_insights.src.report_generator_genai import ReportGeneratorGenAI

# After running simulation
env = FinancialEnvironment(num_banks=20)
# ... run simulation ...

# Generate AI analysis
aggregator = DataAggregatorGenAI(env, env.history)
metrics = aggregator.aggregate()

analyzer = FeatherlessAnalyzer()
ai_analysis = analyzer.analyze(metrics.__dict__)

generator = ReportGeneratorGenAI(metrics.__dict__, ai_analysis)
generator.generate_html_report("outputs/report.html")
```

## 🔍 Systemic Risk Index

The system calculates a composite systemic risk index:

```
Systemic Risk = 0.4 × default_risk + 0.3 × stress_risk + 0.3 × liquidity_risk
```

**Interpretation:**
- **0-20%**: 🟢 Low risk - System healthy
- **20-50%**: 🟡 Moderate - Some stress
- **50-100%**: 🟠 Elevated - Significant risk
- **>100%**: 🔴 Severe/Critical - Crisis conditions

## 🌐 API Integration

The module integrates seamlessly with FastAPI backends:

```python
@app.get("/analysis/generate")
async def generate_analysis():
    env = get_current_simulation()
    aggregator = DataAggregatorGenAI(env, env.history)
    metrics = aggregator.aggregate()
    
    analyzer = FeatherlessAnalyzer()
    analysis = analyzer.analyze(metrics.__dict__)
    
    return {"metrics": metrics.__dict__, "analysis": analysis}
```

## 📝 Notes

- **Demo Data**: The demo uses hardcoded mock data for testing
- **Production**: Use real `FinancialEnvironment` for dynamic results
- **Fallback**: System generates structured analysis if API unavailable
- **API Limits**: Gracefully handles quota/rate limits

## 🔧 Troubleshooting

### API Quota Exceeded
If you see `429` errors, the system automatically falls back to structured analysis based on metrics.

### Import Errors
Ensure you run from project root:
```bash
cd OblivionOverload_Datazen
python ai_insights/demo_with_genai_analysis.py
```

## 📄 License

Part of the OblivionOverload_Datazen financial simulation framework.
