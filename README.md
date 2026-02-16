# Social Media Sentiment → Stock Price Prediction Pipeline

A system that extracts sentiment from social media (Reddit, Twitter) and predicts stock price movements using machine learning.

## Project Overview

This pipeline demonstrates:
- **Alternative data sourcing** - Reddit/Twitter API integration
- **NLP sentiment analysis** - Multiple models (VADER, FinBERT, GPT)
- **Feature engineering** - Time-series aggregation, technical indicators
- **ML modeling** - Gradient boosting with proper validation
- **Backtesting** - Transaction costs, slippage, realistic assumptions
- **Production code quality** - Testing, logging, configuration management

## Expected Performance Baseline

Based on research:
- Social media sentiment can predict stock movements 1-6 days in advance
- Expected accuracy: 55-65% (coin flip is 50%)
- Target Sharpe ratio: 0.8-1.5 (realistic for sentiment-based strategies)
- Win rate: 52-58%

**Critical**: This is an educational project. Real alpha from sentiment decays quickly as more traders use it.

## Architecture

```
Data Collection → Sentiment Analysis → Feature Engineering → ML Model → Backtesting → Live Trading (optional)
```

### Pipeline Components

1. **Data Ingestion Layer**
   - Reddit API (r/wallstreetbets, r/stocks, r/investing)
   - Twitter API (financial accounts, trending tickers)
   - Stock price data (yfinance, Alpha Vantage)

2. **Sentiment Analysis Engine**
   - VADER (baseline - fast, rule-based)
   - FinBERT (finance-specific transformer)
   - OpenAI GPT-4 (for complex analysis)

3. **Feature Store**
   - Aggregated sentiment scores (hourly, daily)
   - Technical indicators (RSI, MACD, Bollinger Bands)
   - Volume metrics
   - Market regime indicators

4. **ML Pipeline**
   - LightGBM / XGBoost (primary models)
   - Neural networks (LSTM for time-series)
   - Ensemble methods

5. **Backtesting Framework**
   - Transaction cost modeling (0.1% per trade)
   - Slippage simulation
   - Position sizing & risk management
   - Walk-forward optimization

## Project Structure

```
sentiment-trading/
├── README.md
├── requirements.txt
├── config/
│   ├── api_config.yaml
│   └── model_config.yaml
├── data/
│   ├── raw/              # Raw API responses
│   ├── processed/        # Cleaned sentiment data
│   └── prices/           # Stock price data
├── src/
│   ├── data_collection/
│   │   ├── reddit_scraper.py
│   │   ├── twitter_scraper.py
│   │   └── price_fetcher.py
│   ├── sentiment/
│   │   ├── vader_analyzer.py
│   │   ├── finbert_analyzer.py
│   │   └── gpt_analyzer.py
│   ├── features/
│   │   ├── sentiment_aggregator.py
│   │   └── technical_indicators.py
│   ├── models/
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── ensemble.py
│   ├── backtesting/
│   │   ├── backtest_engine.py
│   │   └── performance_metrics.py
│   └── utils/
│       ├── data_validator.py
│       └── logger.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_sentiment_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_model_evaluation.ipynb
├── tests/
│   ├── test_data_collection.py
│   ├── test_sentiment.py
│   └── test_backtesting.py
└── results/
    ├── backtest_results/
    └── model_performance/
```

## Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/sentiment-trading
cd sentiment-trading

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

Create `.env` file:
```
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_secret
REDDIT_USER_AGENT=your_user_agent
TWITTER_BEARER_TOKEN=your_token
ALPHA_VANTAGE_API_KEY=your_key
OPENAI_API_KEY=your_key  # Optional for GPT analysis
```

### 3. Collect Data

```bash
# Collect Reddit data for specific tickers
python src/data_collection/reddit_scraper.py --tickers AAPL TSLA MSFT --days 30

# Fetch corresponding stock prices
python src/data_collection/price_fetcher.py --tickers AAPL TSLA MSFT --days 30
```

### 4. Run Sentiment Analysis

```bash
# Process with multiple sentiment models
python src/sentiment/vader_analyzer.py
python src/sentiment/finbert_analyzer.py
```

### 5. Train Model

```bash
# Train with walk-forward validation
python src/models/train.py --model lightgbm --validation walk_forward
```

### 6. Backtest Strategy

```bash
# Run comprehensive backtest
python src/backtesting/backtest_engine.py --strategy sentiment_momentum --costs 0.001
```

## Key Metrics & Validation

### Sentiment Quality Metrics
- Correlation with next-day returns
- Signal-to-noise ratio
- Sentiment divergence detection

### Model Performance
- Out-of-sample accuracy: Target 55-60%
- Sharpe ratio: Target > 1.0
- Max drawdown: Target < 20%
- Win rate: Target > 52%

### Backtest Realism
- 10 bps transaction costs (0.1%)
- 5 bps slippage
- No look-ahead bias
- 1-day holding period minimum

## Testing Strategy

```bash
# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/test_sentiment.py -v

# Check code coverage
pytest --cov=src tests/
```

## Implementation Details

### Phase 1: Data Collection (Week 1)
- Set up Reddit/Twitter API access
- Build robust scrapers with rate limiting
- Implement data validation
- Create data storage schema

### Phase 2: Sentiment Analysis (Week 2)
- Implement VADER (baseline)
- Fine-tune FinBERT on financial data
- Build sentiment aggregation pipeline
- Validate sentiment signals

### Phase 3: Feature Engineering (Week 1)
- Create time-based sentiment features
- Add technical indicators
- Build feature store
- Handle missing data

### Phase 4: Model Development (Week 2)
- Train baseline models
- Implement cross-validation
- Tune hyperparameters
- Build ensemble

### Phase 5: Backtesting (Week 1)
- Build backtest engine
- Add realistic costs
- Generate performance reports
- Sensitivity analysis

## 🎓 Learning Outcomes

This project demonstrates:
1. **Alternative data expertise** - Non-traditional data sources
2. **NLP/ML skills** - Modern sentiment analysis
3. **Quant rigor** - Proper validation, realistic assumptions
4. **Software engineering** - Production-quality code
5. **Domain knowledge** - Understanding market microstructure

## Important Considerations

### What Makes This Project Impressive to SIG:

**DO:**
- Use walk-forward validation (no future data leakage)
- Model transaction costs realistically
- Document all assumptions clearly
- Show failed experiments (shows rigor)
- Compare multiple sentiment models
- Implement proper position sizing

**DON'T:**
- Cherry-pick time periods
- Ignore transaction costs
- Use look-ahead bias
- Claim unrealistic returns (>100% annually)
- Overfit to specific stocks
- Skip out-of-sample testing

## Expected Results to Show

1. **Sentiment-Price Correlation Analysis**
   - Lead-lag relationships
   - Correlation by time of day
   - Correlation by market regime

2. **Feature Importance**
   - Which sentiment metrics matter most
   - Technical vs. sentiment feature contribution

3. **Strategy Performance**
   - Equity curve with drawdowns
   - Monthly returns distribution
   - Performance by market condition

4. **Sensitivity Analysis**
   - Impact of transaction costs
   - Impact of holding period
   - Impact of position sizing

## Future Enhancements

- Real-time sentiment streaming
- Multi-asset strategy (ETFs, indices)
- News sentiment integration
- Cross-validation across market regimes
- Portfolio optimization

## References

- ["Predicting Stock Market Movement with Social Media Sentiment"](https://research-papers)
- FinBERT: Financial Sentiment Analysis with Pre-trained Language Models
- VADER Sentiment Analysis

## Contributing

This is a portfolio project, but feedback is welcome via issues.

## Disclaimer

This is for educational purposes only. Not financial advice. Past performance does not guarantee future results. Trading involves risk of loss.

## License

MIT License - See LICENSE file for details
