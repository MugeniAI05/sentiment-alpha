"""
Sentiment Analysis Module

Implements multiple sentiment analysis approaches:
1. VADER - Fast, rule-based (baseline)
2. FinBERT - Financial domain-specific transformer
3. Ensemble combining multiple signals
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
from datetime import datetime

# VADER
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# FinBERT (install: pip install transformers torch)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn.functional import softmax

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VADERSentimentAnalyzer:
    """Rule-based sentiment analysis using VADER."""
    
    def __init__(self):
        """Initialize VADER analyzer."""
        self.analyzer = SentimentIntensityAnalyzer()
        logger.info("VADER analyzer initialized")
    
    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment scores
        """
        scores = self.analyzer.polarity_scores(text)
        return {
            'vader_positive': scores['pos'],
            'vader_negative': scores['neg'],
            'vader_neutral': scores['neu'],
            'vader_compound': scores['compound']
        }
    
    def analyze_batch(self, texts: List[str]) -> pd.DataFrame:
        """Analyze multiple texts."""
        results = [self.analyze(text) for text in texts]
        return pd.DataFrame(results)


class FinBERTSentimentAnalyzer:
    """Financial sentiment analysis using FinBERT transformer."""
    
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        Initialize FinBERT model.
        
        Args:
            model_name: HuggingFace model name
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"FinBERT model loaded: {model_name}")
    
    def analyze(self, text: str, max_length: int = 512) -> Dict[str, float]:
        """
        Analyze sentiment using FinBERT.
        
        Args:
            text: Input text
            max_length: Maximum token length
            
        Returns:
            Dictionary with sentiment probabilities
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = softmax(outputs.logits, dim=-1)
        
        # FinBERT outputs: [positive, negative, neutral]
        probs = predictions[0].cpu().numpy()
        
        return {
            'finbert_positive': float(probs[0]),
            'finbert_negative': float(probs[1]),
            'finbert_neutral': float(probs[2]),
            'finbert_compound': float(probs[0] - probs[1])  # Net sentiment
        }
    
    def analyze_batch(
        self, 
        texts: List[str], 
        batch_size: int = 16
    ) -> pd.DataFrame:
        """
        Analyze multiple texts in batches.
        
        Args:
            texts: List of texts
            batch_size: Batch size for processing
            
        Returns:
            DataFrame with sentiment scores
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = softmax(outputs.logits, dim=-1)
            
            # Process each result
            for j, probs in enumerate(predictions.cpu().numpy()):
                results.append({
                    'finbert_positive': float(probs[0]),
                    'finbert_negative': float(probs[1]),
                    'finbert_neutral': float(probs[2]),
                    'finbert_compound': float(probs[0] - probs[1])
                })
            
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        return pd.DataFrame(results)


class EnsembleSentimentAnalyzer:
    """Combines multiple sentiment analyzers."""
    
    def __init__(self, use_finbert: bool = True):
        """
        Initialize ensemble analyzer.
        
        Args:
            use_finbert: Whether to include FinBERT (slower but more accurate)
        """
        self.vader = VADERSentimentAnalyzer()
        self.use_finbert = use_finbert
        
        if use_finbert:
            try:
                self.finbert = FinBERTSentimentAnalyzer()
            except Exception as e:
                logger.warning(f"FinBERT initialization failed: {e}")
                logger.warning("Falling back to VADER only")
                self.use_finbert = False
    
    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = 'text'
    ) -> pd.DataFrame:
        """
        Add sentiment scores to DataFrame.
        
        Args:
            df: Input DataFrame with text
            text_column: Name of text column
            
        Returns:
            DataFrame with added sentiment columns
        """
        logger.info(f"Analyzing sentiment for {len(df)} texts")
        
        # Combine title and body if both exist
        if 'title' in df.columns and 'body' in df.columns:
            texts = (df['title'].fillna('') + ' ' + df['body'].fillna('')).tolist()
        else:
            texts = df[text_column].fillna('').tolist()
        
        # VADER analysis (fast)
        logger.info("Running VADER analysis...")
        vader_scores = self.vader.analyze_batch(texts)
        
        # FinBERT analysis (slower)
        if self.use_finbert:
            logger.info("Running FinBERT analysis...")
            finbert_scores = self.finbert.analyze_batch(texts)
        else:
            finbert_scores = pd.DataFrame()
        
        # Combine scores
        result_df = df.copy()
        result_df = pd.concat([result_df, vader_scores], axis=1)
        
        if not finbert_scores.empty:
            result_df = pd.concat([result_df, finbert_scores], axis=1)
            
            # Create ensemble score (weighted average)
            result_df['ensemble_compound'] = (
                0.4 * result_df['vader_compound'] + 
                0.6 * result_df['finbert_compound']
            )
        else:
            result_df['ensemble_compound'] = result_df['vader_compound']
        
        # Categorize sentiment
        result_df['sentiment_label'] = pd.cut(
            result_df['ensemble_compound'],
            bins=[-1, -0.05, 0.05, 1],
            labels=['negative', 'neutral', 'positive']
        )
        
        logger.info("Sentiment analysis complete")
        
        return result_df


def analyze_reddit_data(
    input_path: str,
    output_path: str,
    use_finbert: bool = True
) -> pd.DataFrame:
    """
    Analyze sentiment of Reddit data.
    
    Args:
        input_path: Path to Reddit CSV
        output_path: Path to save results
        use_finbert: Whether to use FinBERT
        
    Returns:
        DataFrame with sentiment scores
    """
    # Load data
    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    
    # Initialize analyzer
    analyzer = EnsembleSentimentAnalyzer(use_finbert=use_finbert)
    
    # Analyze sentiment
    result_df = analyzer.analyze_dataframe(df)
    
    # Save results
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")
    
    # Print summary
    print("\n=== Sentiment Analysis Summary ===")
    print(f"Total texts analyzed: {len(result_df)}")
    print(f"\nSentiment distribution:")
    print(result_df['sentiment_label'].value_counts())
    print(f"\nAverage sentiment scores:")
    print(f"  VADER compound: {result_df['vader_compound'].mean():.3f}")
    if 'finbert_compound' in result_df.columns:
        print(f"  FinBERT compound: {result_df['finbert_compound'].mean():.3f}")
    print(f"  Ensemble compound: {result_df['ensemble_compound'].mean():.3f}")
    
    return result_df


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze sentiment of social media data')
    parser.add_argument('--input', required=True, help='Input CSV file')
    parser.add_argument('--output', required=True, help='Output CSV file')
    parser.add_argument('--no-finbert', action='store_true', help='Skip FinBERT (faster)')
    
    args = parser.parse_args()
    
    analyze_reddit_data(
        input_path=args.input,
        output_path=args.output,
        use_finbert=not args.no_finbert
    )


if __name__ == "__main__":
    main()
