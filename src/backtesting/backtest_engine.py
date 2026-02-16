"""
Backtesting Engine

Simulates trading strategy with realistic transaction costs, slippage,
and position sizing.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Backtests sentiment-based trading strategies.
    
    Critical assumptions to avoid overfitting:
    - Transaction costs: 0.1% per trade (10 bps)
    - Slippage: 0.05% per trade (5 bps)
    - No look-ahead bias
    - Realistic position sizing
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        transaction_cost: float = 0.001,  # 10 bps
        slippage: float = 0.0005,  # 5 bps
        position_size: float = 0.1  # 10% of portfolio per trade
    ):
        """
        Initialize backtest engine.
        
        Args:
            initial_capital: Starting capital
            transaction_cost: Transaction cost as fraction
            slippage: Slippage as fraction
            position_size: Position size as fraction of portfolio
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.position_size = position_size
        
        self.trades = []
        self.equity_curve = []
    
    def run_backtest(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        prediction_proba: Optional[np.ndarray] = None,
        confidence_threshold: float = 0.6
    ) -> pd.DataFrame:
        """
        Run backtest with predictions.
        
        Args:
            df: DataFrame with prices and returns
            predictions: Binary predictions (0/1)
            prediction_proba: Prediction probabilities (optional)
            confidence_threshold: Minimum confidence to trade
            
        Returns:
            DataFrame with backtest results
        """
        logger.info(f"Running backtest with {len(df)} periods")
        
        # Initialize tracking
        capital = self.initial_capital
        position = 0  # Shares held
        position_value = 0
        
        results = []
        
        for i in range(len(df)):
            row = df.iloc[i]
            pred = predictions[i]
            
            # Get confidence if available
            confidence = prediction_proba[i] if prediction_proba is not None else 1.0
            
            # Current price
            price = row['Close']
            
            # Calculate current portfolio value
            portfolio_value = capital + (position * price)
            
            # Trading logic
            action = None
            trade_cost = 0
            
            # Use confidence threshold to filter trades
            if confidence < confidence_threshold:
                pred = 0  # Don't trade if not confident
            
            # LONG signal and no position
            if pred == 1 and position == 0:
                # Buy
                shares_to_buy = int((portfolio_value * self.position_size) / price)
                if shares_to_buy > 0:
                    cost = shares_to_buy * price
                    trade_cost = cost * (self.transaction_cost + self.slippage)
                    
                    if capital >= (cost + trade_cost):
                        position = shares_to_buy
                        position_value = cost
                        capital -= (cost + trade_cost)
                        action = 'BUY'
                        
                        self.trades.append({
                            'timestamp': row['timestamp'],
                            'action': action,
                            'price': price,
                            'shares': shares_to_buy,
                            'cost': trade_cost,
                            'confidence': confidence
                        })
            
            # Exit signal or stop loss
            elif pred == 0 and position > 0:
                # Sell
                revenue = position * price
                trade_cost = revenue * (self.transaction_cost + self.slippage)
                
                capital += (revenue - trade_cost)
                pnl = revenue - position_value - (2 * position_value * (self.transaction_cost + self.slippage))
                
                action = 'SELL'
                
                self.trades.append({
                    'timestamp': row['timestamp'],
                    'action': action,
                    'price': price,
                    'shares': position,
                    'cost': trade_cost,
                    'pnl': pnl,
                    'confidence': confidence
                })
                
                position = 0
                position_value = 0
            
            # Record equity
            current_value = capital + (position * price)
            
            results.append({
                'timestamp': row['timestamp'],
                'price': price,
                'prediction': pred,
                'confidence': confidence,
                'position': position,
                'capital': capital,
                'portfolio_value': current_value,
                'action': action if action else 'HOLD'
            })
        
        results_df = pd.DataFrame(results)
        
        logger.info(f"Backtest complete: {len(self.trades)} trades executed")
        
        return results_df
    
    def calculate_metrics(self, results_df: pd.DataFrame) -> Dict:
        """Calculate performance metrics."""
        
        # Returns
        results_df['returns'] = results_df['portfolio_value'].pct_change()
        
        # Total return
        total_return = (
            (results_df['portfolio_value'].iloc[-1] - self.initial_capital) 
            / self.initial_capital
        )
        
        # Annualized return (assuming hourly data, 252 trading days, 6.5 hours/day)
        hours_per_year = 252 * 6.5
        periods = len(results_df)
        years = periods / hours_per_year
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Volatility
        volatility = results_df['returns'].std() * np.sqrt(hours_per_year)
        
        # Sharpe ratio (assuming 2% risk-free rate)
        sharpe_ratio = (annualized_return - 0.02) / volatility if volatility > 0 else 0
        
        # Max drawdown
        cumulative = (1 + results_df['returns']).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        trades_df = pd.DataFrame(self.trades)
        if not trades_df.empty and 'pnl' in trades_df.columns:
            winning_trades = (trades_df['pnl'] > 0).sum()
            total_trades = len(trades_df[trades_df['action'] == 'SELL'])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
        else:
            win_rate = 0
            total_trades = 0
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'final_value': results_df['portfolio_value'].iloc[-1]
        }
        
        return metrics
    
    def print_results(self, metrics: Dict):
        """Print backtest results."""
        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)
        print(f"Initial Capital:      ${self.initial_capital:,.2f}")
        print(f"Final Value:          ${metrics['final_value']:,.2f}")
        print(f"Total Return:         {metrics['total_return']:.2%}")
        print(f"Annualized Return:    {metrics['annualized_return']:.2%}")
        print(f"Volatility:           {metrics['volatility']:.2%}")
        print(f"Sharpe Ratio:         {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown:         {metrics['max_drawdown']:.2%}")
        print(f"Win Rate:             {metrics['win_rate']:.2%}")
        print(f"Total Trades:         {metrics['total_trades']}")
        print(f"Transaction Cost:     {self.transaction_cost:.2%}")
        print(f"Slippage:             {self.slippage:.2%}")
        print("="*50 + "\n")
    
    def plot_results(self, results_df: pd.DataFrame, save_path: Optional[str] = None):
        """Plot backtest results."""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # 1. Equity curve
        axes[0].plot(results_df['timestamp'], results_df['portfolio_value'], label='Portfolio Value')
        axes[0].axhline(y=self.initial_capital, color='r', linestyle='--', label='Initial Capital')
        axes[0].set_title('Portfolio Value Over Time')
        axes[0].set_ylabel('Value ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Drawdown
        returns = results_df['portfolio_value'].pct_change()
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        
        axes[1].fill_between(results_df['timestamp'], 0, drawdown * 100, alpha=0.3, color='red')
        axes[1].set_title('Drawdown')
        axes[1].set_ylabel('Drawdown (%)')
        axes[1].grid(True, alpha=0.3)
        
        # 3. Trade signals
        buy_signals = results_df[results_df['action'] == 'BUY']
        sell_signals = results_df[results_df['action'] == 'SELL']
        
        axes[2].plot(results_df['timestamp'], results_df['price'], label='Price', alpha=0.7)
        axes[2].scatter(buy_signals['timestamp'], buy_signals['price'], 
                       color='green', marker='^', s=100, label='Buy', zorder=5)
        axes[2].scatter(sell_signals['timestamp'], sell_signals['price'],
                       color='red', marker='v', s=100, label='Sell', zorder=5)
        axes[2].set_title('Trade Signals')
        axes[2].set_ylabel('Price ($)')
        axes[2].set_xlabel('Date')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()


def main():
    """Example usage."""
    import argparse
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from models.train import ModelTrainer
    
    parser = argparse.ArgumentParser(description='Backtest trading strategy')
    parser.add_argument('--features', required=True, help='Feature CSV')
    parser.add_argument('--model', required=True, help='Model directory')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital')
    parser.add_argument('--transaction-cost', type=float, default=0.001, help='Transaction cost')
    parser.add_argument('--confidence', type=float, default=0.6, help='Confidence threshold')
    parser.add_argument('--output', default='results/backtest', help='Output directory')
    
    args = parser.parse_args()
    
    # Load model
    logger.info("Loading model...")
    trainer = ModelTrainer.load_model(args.model)
    
    # Load features
    logger.info("Loading features...")
    df = pd.read_csv(args.features)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Prepare features
    X, y, feature_names = trainer.prepare_data(df)
    
    # Make predictions
    logger.info("Making predictions...")
    predictions = trainer.predict(X)
    prediction_proba = trainer.predict_proba(X)[:, 1]
    
    # Align with DataFrame
    df_aligned = df.iloc[X.index].reset_index(drop=True)
    
    # Run backtest
    engine = BacktestEngine(
        initial_capital=args.capital,
        transaction_cost=args.transaction_cost
    )
    
    results_df = engine.run_backtest(
        df_aligned,
        predictions,
        prediction_proba,
        confidence_threshold=args.confidence
    )
    
    # Calculate metrics
    metrics = engine.calculate_metrics(results_df)
    engine.print_results(metrics)
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_df.to_csv(output_dir / 'backtest_results.csv', index=False)
    
    # Plot
    engine.plot_results(results_df, save_path=output_dir / 'backtest_plot.png')
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
