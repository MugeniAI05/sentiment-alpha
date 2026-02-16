"""
Machine Learning Model Training Module

Implements rigorous training with walk-forward validation to avoid overfitting.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path
from datetime import datetime
import joblib
import json

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report
)

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains ML models with proper time-series validation."""
    
    # Feature groups for easier selection
    SENTIMENT_FEATURES = [
        'sentiment_ensemble_compound_mean',
        'sentiment_ensemble_compound_std',
        'sentiment_momentum_1h',
        'sentiment_momentum_4h',
        'sentiment_momentum_24h',
        'sentiment_volatility_4h',
        'mention_volume_4h',
        'mention_volume_24h',
        'engagement_score'
    ]
    
    TECHNICAL_FEATURES = [
        'returns',
        'rsi',
        'macd',
        'macd_histogram',
        'bb_position',
        'bb_width',
        'volatility_10',
        'volume_ratio',
        'momentum_10',
        'atr'
    ]
    
    def __init__(self, model_type: str = 'lightgbm'):
        """
        Initialize trainer.
        
        Args:
            model_type: 'lightgbm', 'xgboost', or 'random_forest'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = 'target_direction_24h',
        feature_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare data for training.
        
        Args:
            df: Feature DataFrame
            target_col: Target column name
            feature_cols: List of feature columns (None = auto-select)
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        # Auto-select features if not provided
        if feature_cols is None:
            feature_cols = self.SENTIMENT_FEATURES + self.TECHNICAL_FEATURES
        
        # Filter to available features
        available_features = [f for f in feature_cols if f in df.columns]
        
        if len(available_features) < len(feature_cols):
            missing = set(feature_cols) - set(available_features)
            logger.warning(f"Missing features: {missing}")
        
        logger.info(f"Using {len(available_features)} features")
        
        # Extract features and target
        X = df[available_features].copy()
        y = df[target_col].copy()
        
        # Remove rows with NaN
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_idx]
        y = y[valid_idx]
        
        logger.info(f"Data shape: {X.shape}, Target distribution: {y.value_counts().to_dict()}")
        
        return X, y, available_features
    
    def walk_forward_validation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        test_size: int = None
    ) -> Dict[str, List[float]]:
        """
        Perform walk-forward validation.
        
        Args:
            X: Features
            y: Target
            n_splits: Number of splits
            test_size: Size of test set (None = auto)
            
        Returns:
            Dictionary of metrics across folds
        """
        logger.info(f"Starting walk-forward validation with {n_splits} splits")
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        
        metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'roc_auc': []
        }
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            logger.info(f"Fold {fold}/{n_splits}")
            
            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            model = self._create_model()
            model.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            metrics['accuracy'].append(accuracy_score(y_test, y_pred))
            metrics['precision'].append(precision_score(y_test, y_pred, zero_division=0))
            metrics['recall'].append(recall_score(y_test, y_pred, zero_division=0))
            metrics['f1'].append(f1_score(y_test, y_pred, zero_division=0))
            metrics['roc_auc'].append(roc_auc_score(y_test, y_pred_proba))
            
            logger.info(f"  Accuracy: {metrics['accuracy'][-1]:.4f}")
            logger.info(f"  ROC-AUC: {metrics['roc_auc'][-1]:.4f}")
        
        # Print summary
        print("\n=== Walk-Forward Validation Results ===")
        for metric_name, values in metrics.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{metric_name}: {mean_val:.4f} (+/- {std_val:.4f})")
        
        return metrics
    
    def train_final_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.2
    ):
        """
        Train final model on all available data.
        
        Args:
            X: Features
            y: Target
            validation_split: Fraction for validation set
        """
        logger.info("Training final model")
        
        # Split into train and validation (time-based)
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train
        self.model = self._create_model()
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate on validation set
        y_pred = self.model.predict(X_val_scaled)
        y_pred_proba = self.model.predict_proba(X_val_scaled)[:, 1]
        
        print("\n=== Final Model Performance (Validation Set) ===")
        print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
        print(f"Precision: {precision_score(y_val, y_pred):.4f}")
        print(f"Recall: {recall_score(y_val, y_pred):.4f}")
        print(f"F1 Score: {f1_score(y_val, y_pred):.4f}")
        print(f"ROC-AUC: {roc_auc_score(y_val, y_pred_proba):.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred, target_names=['Down', 'Up']))
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Features:")
            print(self.feature_importance.head(10).to_string(index=False))
    
    def _create_model(self):
        """Create model instance based on type."""
        if self.model_type == 'lightgbm':
            return lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                num_leaves=31,
                random_state=42,
                verbose=-1
            )
        elif self.model_type == 'xgboost':
            return xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                random_state=42
            )
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def save_model(self, output_dir: str):
        """Save trained model and artifacts."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_file = output_path / f"{self.model_type}_model.pkl"
        joblib.dump(self.model, model_file)
        logger.info(f"Model saved to {model_file}")
        
        # Save scaler
        scaler_file = output_path / "scaler.pkl"
        joblib.dump(self.scaler, scaler_file)
        
        # Save feature importance
        if self.feature_importance is not None:
            importance_file = output_path / "feature_importance.csv"
            self.feature_importance.to_csv(importance_file, index=False)
        
        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'trained_at': datetime.now().isoformat(),
            'feature_count': len(self.scaler.mean_) if hasattr(self.scaler, 'mean_') else None
        }
        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"All artifacts saved to {output_dir}")
    
    @classmethod
    def load_model(cls, model_dir: str):
        """Load trained model."""
        model_path = Path(model_dir)
        
        # Load metadata
        with open(model_path / "metadata.json") as f:
            metadata = json.load(f)
        
        # Create trainer
        trainer = cls(model_type=metadata['model_type'])
        
        # Load model and scaler
        trainer.model = joblib.load(model_path / f"{metadata['model_type']}_model.pkl")
        trainer.scaler = joblib.load(model_path / "scaler.pkl")
        
        # Load feature importance
        importance_file = model_path / "feature_importance.csv"
        if importance_file.exists():
            trainer.feature_importance = pd.read_csv(importance_file)
        
        logger.info(f"Model loaded from {model_dir}")
        
        return trainer
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained. Call train_final_model() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        if self.model is None:
            raise ValueError("Model not trained. Call train_final_model() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train sentiment-based trading model')
    parser.add_argument('--features', required=True, help='Feature CSV file')
    parser.add_argument('--model', default='lightgbm', choices=['lightgbm', 'xgboost', 'random_forest'])
    parser.add_argument('--target', default='target_direction_24h', help='Target column')
    parser.add_argument('--output', default='models/sentiment_model', help='Output directory')
    parser.add_argument('--validate', action='store_true', help='Run walk-forward validation')
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading features from {args.features}")
    df = pd.read_csv(args.features)
    
    # Initialize trainer
    trainer = ModelTrainer(model_type=args.model)
    
    # Prepare data
    X, y, feature_names = trainer.prepare_data(df, target_col=args.target)
    
    # Walk-forward validation (optional)
    if args.validate:
        metrics = trainer.walk_forward_validation(X, y, n_splits=5)
    
    # Train final model
    trainer.train_final_model(X, y, validation_split=0.2)
    
    # Save
    trainer.save_model(args.output)
    
    print(f"\nTraining complete! Model saved to {args.output}")


if __name__ == "__main__":
    main()
