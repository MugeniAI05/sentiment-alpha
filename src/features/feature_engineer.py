"""
Feature Engineering Module

Combines sentiment signals with technical indicators and price data
to create ML-ready features.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
import logging
from datetime import datetime, timedelta
import yfinance as yf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentAggregator:
    """Aggregates sentiment data into time-based features."""
    
    @staticmethod
    def aggregate_sentiment(
        df: pd.DataFrame,
        ticker: str,
        freq: str = '1H'
    ) -> pd.DataFrame:
        """
        Aggregate sentiment by time period.
        
        Args:
            df: DataFrame with sentiment scores
            ticker: Ticker symbol to filter
            freq: Aggregation frequency ('1H', '4H', '1D')
            
        Returns:
            Aggregated DataFrame
        """
        # Filter to specific ticker
        ticker_df = df[df['tickers'].str.contains(ticker, na=False)].copy()
        
        if ticker_df.empty:
            logger.warning(f"No data found for {ticker}")
            return pd.DataFrame()
        
        # Ensure datetime
        ticker_df['created_utc'] = pd.to_datetime(ticker_df['created_utc'])
        ticker_df.set_index('created_utc', inplace=True)
        
        # Aggregate by time period
        agg_df = ticker_df.resample(freq).agg({
            'ensemble_compound': ['mean', 'std', 'min', 'max', 'count'],
            'vader_compound': ['mean'],
            'score': ['sum', 'mean'],  # Reddit score (upvotes)
            'num_comments': ['sum', 'mean']
        })
        
        # Flatten column names
        agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
        agg_df.columns = [f'sentiment_{col}' for col in agg_df.columns]
        
        # Reset index
        agg_df = agg_df.reset_index()
        agg_df.rename(columns={'created_utc': 'timestamp'}, inplace=True)
        
        # Add ticker
        agg_df['ticker'] = ticker
        
        # Forward fill missing periods (no posts = previous sentiment)
        agg_df = agg_df.fillna(method='ffill')
        
        return agg_df
    
    @staticmethod
    def create_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived sentiment features.
        
        Args:
            df: Aggregated sentiment DataFrame
            
        Returns:
            DataFrame with additional features
        """
        df = df.copy()
        
        # Sentiment momentum (change over time)
        df['sentiment_momentum_1h'] = df['sentiment_ensemble_compound_mean'].diff(1)
        df['sentiment_momentum_4h'] = df['sentiment_ensemble_compound_mean'].diff(4)
        df['sentiment_momentum_24h'] = df['sentiment_ensemble_compound_mean'].diff(24)
        
        # Sentiment volatility (rolling std)
        df['sentiment_volatility_4h'] = df['sentiment_ensemble_compound_mean'].rolling(4).std()
        df['sentiment_volatility_24h'] = df['sentiment_ensemble_compound_mean'].rolling(24).std()
        
        # Volume of mentions (activity)
        df['mention_volume_4h'] = df['sentiment_ensemble_compound_count'].rolling(4).sum()
        df['mention_volume_24h'] = df['sentiment_ensemble_compound_count'].rolling(24).sum()
        
        # Sentiment extremes (max deviation from mean)
        rolling_mean = df['sentiment_ensemble_compound_mean'].rolling(24).mean()
        df['sentiment_extreme_positive'] = (
            df['sentiment_ensemble_compound_max'] - rolling_mean
        ).clip(lower=0)
        df['sentiment_extreme_negative'] = (
            rolling_mean - df['sentiment_ensemble_compound_min']
        ).clip(lower=0)
        
        # Engagement score (upvotes * mentions)
        df['engagement_score'] = (
            df['sentiment_score_sum'] * df['sentiment_ensemble_compound_count']
        )
        
        return df


class TechnicalIndicators:
    """Calculate technical indicators from price data."""
    
    @staticmethod
    def fetch_price_data(
        ticker: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Fetch price data from Yahoo Finance."""
        logger.info(f"Fetching price data for {ticker}")
        
        try:
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval='1h',
                progress=False
            )
            
            if data.empty:
                logger.warning(f"No price data found for {ticker}")
                return pd.DataFrame()
            
            data = data.reset_index()
            data.rename(columns={'Datetime': 'timestamp'}, inplace=True)
            data['ticker'] = ticker
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators.
        
        Args:
            df: Price DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicators
        """
        df = df.copy()
        
        # Returns
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Moving averages
        df['sma_10'] = df['Close'].rolling(10).mean()
        df['sma_50'] = df['Close'].rolling(50).mean()
        df['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (2 * bb_std)
        df['bb_lower'] = df['bb_middle'] - (2 * bb_std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volatility
        df['volatility_10'] = df['returns'].rolling(10).std()
        df['volatility_50'] = df['returns'].rolling(50).std()
        
        # Volume indicators
        df['volume_sma'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']
        
        # Price momentum
        df['momentum_10'] = df['Close'] - df['Close'].shift(10)
        df['momentum_50'] = df['Close'] - df['Close'].shift(50)
        
        # ATR (Average True Range)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        return df


class FeatureEngineer:
    """Combines sentiment and technical features."""
    
    def __init__(self):
        self.sentiment_agg = SentimentAggregator()
        self.tech_indicators = TechnicalIndicators()
    
    def create_features(
        self,
        sentiment_df: pd.DataFrame,
        ticker: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Create complete feature set.
        
        Args:
            sentiment_df: Raw sentiment data
            ticker: Ticker symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            Combined feature DataFrame
        """
        logger.info(f"Creating features for {ticker}")
        
        # 1. Aggregate sentiment
        sentiment_features = self.sentiment_agg.aggregate_sentiment(
            sentiment_df, ticker, freq='1H'
        )
        
        if sentiment_features.empty:
            return pd.DataFrame()
        
        # 2. Add derived sentiment features
        sentiment_features = self.sentiment_agg.create_sentiment_features(
            sentiment_features
        )
        
        # 3. Fetch price data
        price_df = self.tech_indicators.fetch_price_data(
            ticker, start_date, end_date
        )
        
        if price_df.empty:
            return pd.DataFrame()
        
        # 4. Calculate technical indicators
        price_df = self.tech_indicators.calculate_indicators(price_df)
        
        # 5. Merge sentiment and price features
        # Ensure both have timestamp as datetime
        sentiment_features['timestamp'] = pd.to_datetime(sentiment_features['timestamp'])
        price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
        
        # Merge on nearest timestamp (align hourly)
        merged_df = pd.merge_asof(
            price_df.sort_values('timestamp'),
            sentiment_features.sort_values('timestamp'),
            on='timestamp',
            direction='backward',
            suffixes=('', '_sent')
        )
        
        # 6. Create target variable (future returns)
        merged_df['target_1h'] = merged_df['Close'].shift(-1) / merged_df['Close'] - 1
        merged_df['target_4h'] = merged_df['Close'].shift(-4) / merged_df['Close'] - 1
        merged_df['target_24h'] = merged_df['Close'].shift(-24) / merged_df['Close'] - 1
        
        # 7. Create binary classification targets
        merged_df['target_direction_1h'] = (merged_df['target_1h'] > 0).astype(int)
        merged_df['target_direction_4h'] = (merged_df['target_4h'] > 0).astype(int)
        merged_df['target_direction_24h'] = (merged_df['target_24h'] > 0).astype(int)
        
        # 8. Drop rows with NaN in critical columns
        merged_df = merged_df.dropna(subset=['target_1h', 'sentiment_ensemble_compound_mean'])
        
        logger.info(f"Created {len(merged_df)} feature rows for {ticker}")
        
        return merged_df


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create ML features')
    parser.add_argument('--sentiment', required=True, help='Sentiment CSV file')
    parser.add_argument('--ticker', required=True, help='Ticker symbol')
    parser.add_argument('--output', required=True, help='Output CSV file')
    parser.add_argument('--days', type=int, default=30, help='Days of price data')
    
    args = parser.parse_args()
    
    # Load sentiment data
    sentiment_df = pd.read_csv(args.sentiment)
    
    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    # Create features
    engineer = FeatureEngineer()
    features_df = engineer.create_features(
        sentiment_df=sentiment_df,
        ticker=args.ticker,
        start_date=start_date,
        end_date=end_date
    )
    
    # Save
    if not features_df.empty:
        features_df.to_csv(args.output, index=False)
        print(f"\nFeatures saved to {args.output}")
        print(f"Total rows: {len(features_df)}")
        print(f"Total columns: {len(features_df.columns)}")


if __name__ == "__main__":
    main()
