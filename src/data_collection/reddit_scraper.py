"""
Reddit Data Scraper for Sentiment Analysis

Collects posts and comments from financial subreddits mentioning specific tickers.
Implements rate limiting, error handling, and data validation.
"""

import praw
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
from pathlib import Path
import json
import time
from collections import defaultdict
import re
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()


class RedditScraper:
    """Scrapes Reddit for stock-related posts and comments."""
    
    # Financial subreddits to monitor
    SUBREDDITS = [
        'wallstreetbets',
        'stocks', 
        'investing',
        'StockMarket',
        'options',
        'pennystocks',
        'Daytrading'
    ]
    
    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        """Initialize Reddit API connection."""
        self.client_id = client_id or os.getenv('REDDIT_CLIENT_ID')
        self.client_secret = client_secret or os.getenv('REDDIT_CLIENT_SECRET')
        self.user_agent = user_agent or os.getenv('REDDIT_USER_AGENT')
        
        if not all([self.client_id, self.client_secret, self.user_agent]):
            raise ValueError("Reddit API credentials not found. Set environment variables.")
        
        self.reddit = praw.Reddit(
            client_id=self.client_id,
            client_secret=self.client_secret,
            user_agent=self.user_agent
        )
        
        logger.info("Reddit API connection established")
    
    def extract_tickers(self, text: str, valid_tickers: set) -> List[str]:
        """
        Extract stock tickers from text.
        
        Args:
            text: Input text to search
            valid_tickers: Set of valid ticker symbols
            
        Returns:
            List of found ticker symbols
        """
        # Pattern: $ symbol or uppercase words 1-5 chars
        pattern = r'\$([A-Z]{1,5})\b|(?<!\w)([A-Z]{1,5})(?!\w)'
        
        # Extract potential tickers
        matches = re.findall(pattern, text.upper())
        potential_tickers = [m[0] or m[1] for m in matches]
        
        # Filter to only valid tickers and remove common false positives
        false_positives = {'I', 'A', 'BE', 'IT', 'ON', 'AT', 'BY', 'TO', 'CEO', 'CFO', 'DD', 'YOLO', 'DD', 'TA', 'FA'}
        found_tickers = [
            t for t in potential_tickers 
            if t in valid_tickers and t not in false_positives
        ]
        
        return list(set(found_tickers))  # Remove duplicates
    
    def scrape_subreddit(
        self,
        subreddit_name: str,
        tickers: List[str],
        days_back: int = 7,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Scrape posts from a specific subreddit.
        
        Args:
            subreddit_name: Name of subreddit
            tickers: List of ticker symbols to track
            days_back: How many days of history to collect
            limit: Maximum posts to fetch
            
        Returns:
            DataFrame with post data
        """
        logger.info(f"Scraping r/{subreddit_name} for tickers: {tickers}")
        
        subreddit = self.reddit.subreddit(subreddit_name)
        ticker_set = set(tickers)
        posts_data = []
        
        # Calculate cutoff time
        cutoff_time = datetime.utcnow() - timedelta(days=days_back)
        
        try:
            # Fetch recent posts
            for post in subreddit.new(limit=limit):
                post_time = datetime.utcfromtimestamp(post.created_utc)
                
                # Skip if too old
                if post_time < cutoff_time:
                    continue
                
                # Extract tickers from title and body
                title_text = post.title
                body_text = post.selftext if hasattr(post, 'selftext') else ""
                full_text = f"{title_text} {body_text}"
                
                found_tickers = self.extract_tickers(full_text, ticker_set)
                
                # Only save posts mentioning our tickers
                if found_tickers:
                    posts_data.append({
                        'post_id': post.id,
                        'subreddit': subreddit_name,
                        'title': title_text,
                        'body': body_text,
                        'author': str(post.author) if post.author else '[deleted]',
                        'created_utc': post_time,
                        'score': post.score,
                        'upvote_ratio': post.upvote_ratio,
                        'num_comments': post.num_comments,
                        'tickers': ','.join(found_tickers),
                        'url': post.url,
                        'post_type': 'submission'
                    })
                
                # Rate limiting
                time.sleep(0.1)
            
            logger.info(f"Scraped {len(posts_data)} posts from r/{subreddit_name}")
            
        except Exception as e:
            logger.error(f"Error scraping r/{subreddit_name}: {e}")
        
        return pd.DataFrame(posts_data)
    
    def scrape_comments(
        self,
        post_id: str,
        tickers: List[str],
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Scrape comments from a specific post.
        
        Args:
            post_id: Reddit post ID
            tickers: List of tickers to track
            limit: Max comments to fetch
            
        Returns:
            DataFrame with comment data
        """
        ticker_set = set(tickers)
        comments_data = []
        
        try:
            submission = self.reddit.submission(id=post_id)
            submission.comments.replace_more(limit=0)  # Flatten comment tree
            
            for comment in submission.comments.list()[:limit]:
                if not hasattr(comment, 'body'):
                    continue
                
                found_tickers = self.extract_tickers(comment.body, ticker_set)
                
                if found_tickers:
                    comments_data.append({
                        'comment_id': comment.id,
                        'post_id': post_id,
                        'body': comment.body,
                        'author': str(comment.author) if comment.author else '[deleted]',
                        'created_utc': datetime.utcfromtimestamp(comment.created_utc),
                        'score': comment.score,
                        'tickers': ','.join(found_tickers),
                        'post_type': 'comment'
                    })
        
        except Exception as e:
            logger.error(f"Error scraping comments for {post_id}: {e}")
        
        return pd.DataFrame(comments_data)
    
    def scrape_all(
        self,
        tickers: List[str],
        days_back: int = 7,
        include_comments: bool = True,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Scrape all configured subreddits.
        
        Args:
            tickers: List of ticker symbols
            days_back: Days of history to collect
            include_comments: Whether to scrape comments
            save_path: Path to save results
            
        Returns:
            Combined DataFrame
        """
        logger.info(f"Starting full scrape for {len(tickers)} tickers, {days_back} days back")
        
        all_posts = []
        all_comments = []
        
        # Scrape each subreddit
        for subreddit in self.SUBREDDITS:
            posts_df = self.scrape_subreddit(subreddit, tickers, days_back)
            all_posts.append(posts_df)
            
            # Optionally scrape comments
            if include_comments and not posts_df.empty:
                for post_id in posts_df['post_id'].head(50):  # Limit to top 50 posts
                    comments_df = self.scrape_comments(post_id, tickers)
                    all_comments.append(comments_df)
                    time.sleep(0.2)  # Rate limiting
        
        # Combine results
        combined_posts = pd.concat(all_posts, ignore_index=True) if all_posts else pd.DataFrame()
        combined_comments = pd.concat(all_comments, ignore_index=True) if all_comments else pd.DataFrame()
        
        # Merge posts and comments
        if not combined_comments.empty:
            combined_df = pd.concat([combined_posts, combined_comments], ignore_index=True)
        else:
            combined_df = combined_posts
        
        # Save if path provided
        if save_path and not combined_df.empty:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            combined_df.to_csv(save_path, index=False)
            logger.info(f"Saved {len(combined_df)} records to {save_path}")
        
        logger.info(f"Scraping complete: {len(combined_posts)} posts, {len(combined_comments)} comments")
        
        return combined_df
    
    def get_ticker_mentions(self, df: pd.DataFrame) -> Dict[str, int]:
        """Count mentions by ticker."""
        ticker_counts = defaultdict(int)
        
        for tickers_str in df['tickers']:
            for ticker in tickers_str.split(','):
                ticker_counts[ticker] += 1
        
        return dict(sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True))


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Scrape Reddit for stock sentiment')
    parser.add_argument('--tickers', nargs='+', required=True, help='Ticker symbols to track')
    parser.add_argument('--days', type=int, default=7, help='Days of history')
    parser.add_argument('--output', default='data/raw/reddit_data.csv', help='Output file')
    parser.add_argument('--no-comments', action='store_true', help='Skip comment scraping')
    
    args = parser.parse_args()
    
    # Initialize scraper
    scraper = RedditScraper()
    
    # Scrape data
    df = scraper.scrape_all(
        tickers=args.tickers,
        days_back=args.days,
        include_comments=not args.no_comments,
        save_path=args.output
    )
    
    # Print summary
    if not df.empty:
        ticker_counts = scraper.get_ticker_mentions(df)
        print("\n=== Scraping Summary ===")
        print(f"Total records: {len(df)}")
        print(f"\nMentions by ticker:")
        for ticker, count in ticker_counts.items():
            print(f"  {ticker}: {count}")
    else:
        print("No data collected")


if __name__ == "__main__":
    main()
