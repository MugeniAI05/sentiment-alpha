# Quick Reference

## One-Command Setup
```bash
# Clone and setup
git clone https://github.com/yourusername/sentiment-trading
cd sentiment-trading
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Configure APIs (edit .env)
cp .env.example .env
nano .env
```

## Complete Pipeline (Copy-Paste Ready)

### 1. Data Collection (30 min)
```bash
python src/data_collection/reddit_scraper.py \
    --tickers AAPL TSLA NVDA \
    --days 30 \
    --output data/raw/reddit_data.csv
```

### 2. Sentiment Analysis (2-3 hours with FinBERT)
```bash
# Fast version (VADER only)
python src/sentiment/sentiment_analyzer.py \
    --input data/raw/reddit_data.csv \
    --output data/processed/sentiment.csv \
    --no-finbert

# Accurate version (includes FinBERT)
python src/sentiment/sentiment_analyzer.py \
    --input data/raw/reddit_data.csv \
    --output data/processed/sentiment.csv
```

### 3. Feature Engineering (15 min per ticker)
```bash
python src/features/feature_engineer.py \
    --sentiment data/processed/sentiment.csv \
    --ticker AAPL \
    --days 30 \
    --output data/features/AAPL_features.csv
```

### 4. Model Training (30 min)
```bash
# With validation
python src/models/train.py \
    --features data/features/AAPL_features.csv \
    --model lightgbm \
    --validate \
    --output models/sentiment_model
```

### 5. Backtesting (5 min)
```bash
python src/backtesting/backtest_engine.py \
    --features data/features/AAPL_features.csv \
    --model models/sentiment_model \
    --output results/backtest
```

## Key Files for Interviews

1. **src/data_collection/reddit_scraper.py** - Shows data engineering skills
2. **src/sentiment/sentiment_analyzer.py** - NLP implementation
3. **src/models/train.py** - ML best practices, validation
4. **src/backtesting/backtest_engine.py** - Trading logic, risk management

## Common Questions & Answers

**Q: Why sentiment analysis?**
A: Social media sentiment can predict stock movements 1-6 days ahead with 55-65% accuracy. Research shows hedge funds using alternative data achieve 3% higher annual returns.

**Q: How do you avoid overfitting?**
A: 1) Walk-forward validation (TimeSeriesSplit), 2) Transaction costs (10 bps), 3) Out-of-sample testing, 4) Feature importance analysis to check if model relies on sentiment vs just technical indicators.

**Q: What's the biggest challenge?**
A: Feature engineering. Raw sentiment has weak correlation (~0.1) with returns. Needed to create derived features like sentiment momentum, mention volume, and engagement scores to capture meaningful signals.

**Q: Real-world applicability?**
A: Sentiment alpha decays quickly as more traders use it. This is educational—production would need real-time streaming, multiple data sources, and constant retraining.

**Q: Next improvements?**
A: 1) Add Twitter/news data, 2) Real-time streaming pipeline, 3) Multi-asset portfolio optimization, 4) Regime detection to avoid volatile periods.

## Performance Benchmarks

### Expected Results (Realistic)
- Accuracy: 55-60% (vs 50% random)
- ROC-AUC: 0.55-0.65
- Sharpe Ratio: 0.8-1.5
- Win Rate: 52-58%
- Annual Return: 5-15% (vs buy-and-hold)

### Red Flags (Indicates Problems)
- Accuracy > 70% → Overfitting
- Sharpe > 3 → Data leakage
- Win rate > 75% → Look-ahead bias
- No consideration of costs → Unrealistic

## Tech Stack Summary

**Data:** Reddit API (PRAW), yfinance
**NLP:** VADER, FinBERT (transformers)
**ML:** LightGBM, XGBoost, scikit-learn
**Backtesting:** Custom engine with realistic costs
**Viz:** matplotlib, seaborn, plotly

## GitHub README Template

```markdown
# Social Media Sentiment Trading Strategy

End-to-end ML pipeline predicting stock movements from Reddit sentiment.

## Key Features
- Multi-model sentiment analysis (VADER + FinBERT)
- Walk-forward validation to prevent overfitting
- Realistic backtesting with transaction costs & slippage
- Feature importance analysis

## Results
- Accuracy: 58% (vs 50% baseline)
- Sharpe Ratio: 1.2
- Outperformed buy-and-hold by 7% annually

## Installation
[setup instructions here]

## Usage
[pipeline commands here]

## Methodology
[approach here]
```



## File Structure Cheat Sheet

```
sentiment-trading/
├── README.md                          # Project overview
├── IMPLEMENTATION_GUIDE.md            # Detailed guide
├── requirements.txt                   # Dependencies
├── .env                              # API keys (gitignored)
├── src/
│   ├── data_collection/
│   │   └── reddit_scraper.py         # ~300 lines
│   ├── sentiment/
│   │   └── sentiment_analyzer.py     # ~250 lines
│   ├── features/
│   │   └── feature_engineer.py       # ~350 lines
│   ├── models/
│   │   └── train.py                  # ~400 lines
│   └── backtesting/
│       └── backtest_engine.py        # ~350 lines
├── notebooks/                         # Jupyter exploration
├── tests/                            # Unit tests
└── results/                          # Backtest outputs
```

## Quick Debug Commands

```bash
# Check data quality
python -c "import pandas as pd; df=pd.read_csv('data/raw/reddit_data.csv'); print(df.info())"

# Verify sentiment distribution
python -c "import pandas as pd; df=pd.read_csv('data/processed/sentiment.csv'); print(df['sentiment_label'].value_counts())"

# Check feature correlations
python -c "import pandas as pd; df=pd.read_csv('data/features/AAPL_features.csv'); print(df[['sentiment_ensemble_compound_mean', 'target_24h']].corr())"
```
