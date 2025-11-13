#!/usr/bin/env python3
"""
ENHANCED Ultimate GUI Forex Trading Bot
Fully automated AI-powered trading with advanced real-time market analysis.
NEW FEATURES:
- Multi-source sentiment analysis (News, Twitter, Reddit)
- Real-time correlation analysis
- Automated withdrawal preparation (with detailed M-Pesa/Bank instructions)
- Enhanced predictive analytics
- Multi-timeframe consensus with ML weighting
- Automated portfolio rebalancing
"""

import time
import json
import math
import logging
import requests
import smtplib
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timezone, timedelta
import argparse
from typing import Dict, Any, Optional, Tuple, List
import sqlite3
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import hashlib
import hmac
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Optional AI imports
try:
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    from textblob import TextBlob
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    logging.warning("AI libraries not available. Install scikit-learn, joblib, and textblob for AI features.")

# -------------------------
# Logging setup
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("trading_bot.log")
    ]
)
logger = logging.getLogger("fx-bot")

# -------------------------
# Database setup (ENHANCED)
# -------------------------
def init_database():
    """Initialize SQLite database with enhanced tables."""
    conn = sqlite3.connect('trades.db')
    c = conn.cursor()
    
    # Existing tables
    c.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            instrument TEXT NOT NULL,
            direction TEXT NOT NULL,
            entry_price REAL NOT NULL,
            exit_price REAL,
            units INTEGER NOT NULL,
            stop_loss REAL,
            take_profit REAL,
            pnl REAL,
            status TEXT NOT NULL,
            exit_reason TEXT,
            duration_minutes INTEGER,
            indicators TEXT,
            ai_signal TEXT,
            ai_confidence REAL,
            sentiment_score REAL,
            correlation_score REAL
        )
    ''')
    
    # NEW: Market sentiment tracking
    c.execute('''
        CREATE TABLE IF NOT EXISTS market_sentiment (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            instrument TEXT NOT NULL,
            news_sentiment REAL,
            social_sentiment REAL,
            combined_sentiment REAL,
            sentiment_sources TEXT,
            volatility_index REAL
        )
    ''')
    
    # NEW: Correlation tracking
    c.execute('''
        CREATE TABLE IF NOT EXISTS correlations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            instrument_pair TEXT NOT NULL,
            correlation_value REAL,
            timeframe TEXT
        )
    ''')
    
    # NEW: Withdrawal tracking
    c.execute('''
        CREATE TABLE IF NOT EXISTS withdrawals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            amount REAL NOT NULL,
            status TEXT NOT NULL,
            withdrawal_method TEXT,
            transaction_details TEXT,
            confirmed BOOLEAN DEFAULT 0
        )
    ''')
    
    # NEW: Advanced market analysis
    c.execute('''
        CREATE TABLE IF NOT EXISTS market_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            instrument TEXT NOT NULL,
            analysis_type TEXT,
            signal_strength REAL,
            confidence REAL,
            supporting_factors TEXT,
            risk_level TEXT
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            starting_balance REAL NOT NULL,
            ending_balance REAL NOT NULL,
            daily_pnl REAL NOT NULL,
            trades_count INTEGER NOT NULL,
            win_rate REAL,
            sharpe_ratio REAL,
            max_drawdown REAL
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            instrument TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            signal_type TEXT NOT NULL,
            indicators TEXT,
            executed BOOLEAN DEFAULT 0
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS ai_learning (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            features TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            actual_outcome TEXT,
            correct BOOLEAN
        )
    ''')
    
    conn.commit()
    conn.close()

init_database()

# -------------------------
# Config loading (ENHANCED)
# -------------------------
def load_config(config_path: str = "config.json"):
    """Load enhanced configuration."""
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        
        required = ["account_id", "api_key"]
        for field in required:
            if field not in config:
                raise ValueError(f"Missing required config field: {field}")
        
        # Set defaults (existing)
        config.setdefault("oanda_api_url", "https://api-fxpractice.oanda.com")
        config.setdefault("instrument", "EUR_USD")
        config.setdefault("granularity", "M5")
        config.setdefault("candle_count", 500)
        config.setdefault("poll_seconds", 30)
        config.setdefault("risk_percent", 0.01)
        config.setdefault("max_daily_loss", 0.05)
        config.setdefault("leverage", 20)
        
        # Indicator settings
        config.setdefault("ema_fast", 12)
        config.setdefault("ema_slow", 26)
        config.setdefault("atr_period", 14)
        config.setdefault("rsi_period", 14)
        config.setdefault("rsi_overbought", 70)
        config.setdefault("rsi_oversold", 30)
        config.setdefault("macd_fast", 12)
        config.setdefault("macd_slow", 26)
        config.setdefault("macd_signal", 9)
        config.setdefault("bb_period", 20)
        config.setdefault("bb_std", 2)
        config.setdefault("atr_multiplier_sl", 2.0)
        config.setdefault("atr_multiplier_tp", 3.0)
        
        # Trailing Stop
        config.setdefault("enable_trailing_stop", True)
        config.setdefault("trailing_stop_atr_multiplier", 1.5)
        config.setdefault("trailing_stop_min_profit_pips", 10)
        
        # AI settings
        config.setdefault("enable_ai_model", True)
        config.setdefault("ai_model_path", "forex_ai_model.pkl")
        config.setdefault("ai_scaler_path", "forex_ai_scaler.pkl")
        config.setdefault("ai_prediction_horizon", 5)
        config.setdefault("ai_retrain_interval_hours", 12)
        config.setdefault("ai_min_confidence", 0.65)
        config.setdefault("ai_learning_rate", 0.1)
        config.setdefault("ai_continuous_learning", True)
        
        # Multi-timeframe
        config.setdefault("enable_multi_timeframe", True)
        config.setdefault("timeframes", ["M5", "M15", "H1"])
        
        # Alerts
        config.setdefault("enable_email_alerts", False)
        config.setdefault("alert_email", "")
        config.setdefault("smtp_server", "smtp.gmail.com")
        config.setdefault("smtp_port", 587)
        config.setdefault("smtp_username", "")
        config.setdefault("smtp_password", "")
        
        config.setdefault("enable_telegram", False)
        config.setdefault("telegram_token", "")
        config.setdefault("telegram_chat_id", "")
        
        # Dashboard
        config.setdefault("enable_dashboard", True)
        config.setdefault("dashboard_port", 8080)
        
        # News Sentiment
        config.setdefault("enable_news_sentiment", True)
        config.setdefault("news_api_key", "YOUR_NEWS_API_KEY")
        
        # NEW: Enhanced Sentiment Analysis
        config.setdefault("enable_social_sentiment", True)
        config.setdefault("twitter_bearer_token", "YOUR_TWITTER_BEARER_TOKEN")
        config.setdefault("reddit_client_id", "YOUR_REDDIT_CLIENT_ID")
        config.setdefault("reddit_client_secret", "YOUR_REDDIT_CLIENT_SECRET")
        config.setdefault("sentiment_weight_news", 0.4)
        config.setdefault("sentiment_weight_social", 0.3)
        config.setdefault("sentiment_weight_technical", 0.3)
        
        # NEW: Correlation Analysis
        config.setdefault("enable_correlation_analysis", True)
        config.setdefault("correlation_instruments", ["EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF"])
        config.setdefault("correlation_threshold", 0.7)
        
        # NEW: Auto-Withdrawal Settings
        config.setdefault("enable_auto_withdrawal_prep", True)
        config.setdefault("min_profit_for_withdrawal", 100.0)
        config.setdefault("withdrawal_percentage", 0.80)
        config.setdefault("mpesa_phone", "")  # Format: +254XXXXXXXXX
        config.setdefault("bank_account", "")
        config.setdefault("bank_name", "")
        
        # NEW: Advanced Risk Management
        config.setdefault("enable_dynamic_position_sizing", True)
        config.setdefault("max_correlation_exposure", 0.75)
        config.setdefault("volatility_adjustment", True)
        config.setdefault("kelly_criterion", False)
        
        # NEW: Market Analysis
        config.setdefault("enable_volume_analysis", True)
        config.setdefault("enable_order_flow_analysis", True)
        config.setdefault("market_depth_levels", 5)
        
        config.setdefault("ignore_incomplete_candles", True)
        
        config.setdefault("backtesting", {
            "enabled": False,
            "start_date": "2023-01-01",
            "end_date": "2023-06-30",
            "initial_balance": 10000.0,
            "plot_equity_curve": True
        })
        
        return config
    except FileNotFoundError:
        logger.error(f"{config_path} not found. Creating template...")
        create_enhanced_config_template(config_path)
        raise SystemExit(f"Please edit {config_path} with your settings")

def create_enhanced_config_template(config_path: str):
    """Create enhanced config template."""
    template = {
        "account_id": "YOUR_ACCOUNT_ID",
        "api_key": "YOUR_API_KEY",
        "oanda_api_url": "https://api-fxpractice.oanda.com",
        "instrument": "EUR_USD",
        "granularity": "M5",
        "candle_count": 500,
        "poll_seconds": 30,
        "risk_percent": 0.01,
        "max_daily_loss": 0.05,
        "leverage": 20,
        
        # Technical Indicators
        "ema_fast": 12,
        "ema_slow": 26,
        "atr_period": 14,
        "rsi_period": 14,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "bb_period": 20,
        "bb_std": 2,
        "atr_multiplier_sl": 2.0,
        "atr_multiplier_tp": 3.0,
        
        # Stop Loss
        "enable_trailing_stop": True,
        "trailing_stop_atr_multiplier": 1.5,
        "trailing_stop_min_profit_pips": 10,
        
        # AI Model
        "enable_ai_model": True,
        "ai_model_path": "forex_ai_model.pkl",
        "ai_scaler_path": "forex_ai_scaler.pkl",
        "ai_prediction_horizon": 5,
        "ai_retrain_interval_hours": 12,
        "ai_min_confidence": 0.65,
        "ai_learning_rate": 0.1,
        "ai_continuous_learning": True,
        
        # Multi-Timeframe
        "enable_multi_timeframe": True,
        "timeframes": ["M5", "M15", "H1"],
        
        # Alerts
        "enable_email_alerts": False,
        "alert_email": "your-email@example.com",
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "smtp_username": "your-email@example.com",
        "smtp_password": "your-app-password",
        
        "enable_telegram": False,
        "telegram_token": "YOUR_BOT_TOKEN",
        "telegram_chat_id": "YOUR_CHAT_ID",
        
        # Dashboard
        "enable_dashboard": True,
        "dashboard_port": 8080,
        
        # Sentiment Analysis
        "enable_news_sentiment": True,
        "news_api_key": "YOUR_NEWS_API_KEY",
        "enable_social_sentiment": True,
        "twitter_bearer_token": "YOUR_TWITTER_BEARER_TOKEN",
        "reddit_client_id": "YOUR_REDDIT_CLIENT_ID",
        "reddit_client_secret": "YOUR_REDDIT_CLIENT_SECRET",
        "sentiment_weight_news": 0.4,
        "sentiment_weight_social": 0.3,
        "sentiment_weight_technical": 0.3,
        
        # Correlation Analysis
        "enable_correlation_analysis": True,
        "correlation_instruments": ["EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF"],
        "correlation_threshold": 0.7,
        
        # Auto-Withdrawal (Preparation Only - Manual Execution Required)
        "enable_auto_withdrawal_prep": True,
        "min_profit_for_withdrawal": 100.0,
        "withdrawal_percentage": 0.80,
        "mpesa_phone": "+254XXXXXXXXX",
        "bank_account": "YOUR_BANK_ACCOUNT",
        "bank_name": "YOUR_BANK_NAME",
        
        # Advanced Risk Management
        "enable_dynamic_position_sizing": True,
        "max_correlation_exposure": 0.75,
        "volatility_adjustment": True,
        "kelly_criterion": False,
        
        # Market Analysis
        "enable_volume_analysis": True,
        "enable_order_flow_analysis": True,
        "market_depth_levels": 5,
        
        "ignore_incomplete_candles": True,
        
        "backtesting": {
            "enabled": False,
            "start_date": "2023-01-01",
            "end_date": "2023-06-30",
            "initial_balance": 10000.0,
            "plot_equity_curve": True
        }
    }
    with open(config_path, "w") as f:
        json.dump(template, f, indent=2)

CONFIG = {}

# Global constants (will be set in main())
OANDA_API_URL = ""
ACCOUNT_ID = ""
API_KEY = ""
INSTRUMENT = ""
GRANULARITY = ""
CANDLE_COUNT = 0
POLL_SECONDS = 0
RISK_PERCENT = 0.0
MAX_DAILY_LOSS = 0.0
LEVERAGE = 0
HEADERS = {}

# -------------------------
# Telegram Bot Integration
# -------------------------
class TelegramBot:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.enabled = bool(token and chat_id and token != "YOUR_BOT_TOKEN")
    
    def send_message(self, message: str):
        if not self.enabled:
            return
        try:
            url = f"{self.base_url}/sendMessage"
            data = {"chat_id": self.chat_id, "text": message, "parse_mode": "HTML"}
            resp = requests.post(url, json=data, timeout=10)
            if resp.status_code == 200:
                logger.info("Telegram message sent")
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")

def modify_trade_stop_loss(trade_id: str, new_stop_loss: float):
    """Modify the stop loss for an existing trade."""
    try:
        url = f"{OANDA_API_URL}/v3/accounts/{ACCOUNT_ID}/trades/{trade_id}/orders"
        payload = {
            "stopLoss": {
                "price": f"{new_stop_loss:.5f}",
                "timeInForce": "GTC"
            }
        }
        data = make_api_request(url, method='PUT', json_data=payload)
        logger.info(f"✓ Trailing SL updated: {new_stop_loss:.5f} for trade {trade_id}")
        return data
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 400:
            # This can happen if the new SL is not valid (e.g., too close to the current price)
            logger.debug(f"SL modification for trade {trade_id} rejected by API: {e.response.text}")
        else:
            logger.error(f"Error modifying SL: {e.response.text}")
    except Exception as e:
        logger.error(f"Error modifying SL: {e}")
    return None

def manage_trailing_stop():
    """Manage trailing stop loss for open trades."""
    if not CONFIG.get("enable_trailing_stop"):
        return
    
    if STATE.last_trailing_update and (datetime.now(timezone.utc) - STATE.last_trailing_update).seconds < 60:
        return

    try:
        open_trades = get_open_trades()
        
        for trade in open_trades:
            if trade['instrument'] != INSTRUMENT:
                continue
            
            trade_id = trade['id']
            units = int(trade['currentUnits'])
            direction = "BUY" if units > 0 else "SELL"
            entry_price = float(trade['price'])
            current_sl = float(trade.get('stopLossOrder', {}).get('price', 0))
            
            if current_sl == 0:
                continue
            
            current_price = get_current_price(INSTRUMENT)
            
            if direction == "BUY":
                profit_pips = (current_price - entry_price) * 10000
            else:
                profit_pips = (entry_price - current_price) * 10000
            
            min_profit = CONFIG.get("trailing_stop_min_profit_pips", 10)
            if profit_pips < min_profit:
                continue
            
            df = get_candles(INSTRUMENT, 50, GRANULARITY)
            atr_val = atr(df, CONFIG["atr_period"]).iloc[-1]
            trailing_dist = atr_val * CONFIG.get("trailing_stop_atr_multiplier", 1.5)
            
            if direction == "BUY":
                potential_new_sl = current_price - trailing_dist
                if potential_new_sl > entry_price and potential_new_sl > current_sl:
                    modify_trade_stop_loss(trade_id, potential_new_sl)
                    STATE.last_trailing_update = datetime.now(timezone.utc)
            
            elif direction == "SELL":
                potential_new_sl = current_price + trailing_dist
                if potential_new_sl < entry_price and (current_sl == 0 or potential_new_sl < current_sl):
                    modify_trade_stop_loss(trade_id, potential_new_sl)
                    STATE.last_trailing_update = datetime.now(timezone.utc)
        
    except Exception as e:
        logger.error(f"Error in trailing stop management: {e}")

# -------------------------
# NEW: Enhanced Sentiment Analysis
# -------------------------
class EnhancedSentimentAnalyzer:
    def __init__(self):
        self.news_cache = {}
        self.social_cache = {}
        
    def get_multi_source_sentiment(self, instrument: str) -> Tuple[float, Dict[str, float], List[str]]:
        """Get sentiment from multiple sources with weighting."""
        news_sentiment, news_headlines = self.get_news_sentiment(instrument)
        social_sentiment = self.get_social_sentiment(instrument)
        
        # Weighted combination
        news_weight = CONFIG.get("sentiment_weight_news", 0.4)
        social_weight = CONFIG.get("sentiment_weight_social", 0.3)
        
        combined_sentiment = (news_sentiment * news_weight + 
                            social_sentiment * social_weight) / (news_weight + social_weight)
        
        sentiment_breakdown = {
            "news": news_sentiment,
            "social": social_sentiment,
            "combined": combined_sentiment
        }
        
        # Store in database
        self.store_sentiment(instrument, sentiment_breakdown)
        
        return combined_sentiment, sentiment_breakdown, news_headlines
    
    def get_news_sentiment(self, instrument: str) -> Tuple[float, List[str]]:
        """Enhanced news sentiment analysis."""
        if not CONFIG.get("enable_news_sentiment") or not AI_AVAILABLE:
            return 0.0, []

        api_key = CONFIG.get("news_api_key")
        if not api_key or api_key == "YOUR_NEWS_API_KEY":
            return 0.0, []

        try:
            query = instrument.replace("_", " ") + " forex"
            url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&pageSize=30&apiKey={api_key}"
            
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            
            articles = resp.json().get("articles", [])
            if not articles:
                return 0.0, []

            sentiment_scores = []
            headlines = []
            
            for article in articles:
                text = (article['title'] or "") + " " + (article['description'] or "")
                blob = TextBlob(text)
                
                # Enhanced sentiment scoring with subjectivity weighting
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
                
                # Weight sentiment by subjectivity (more subjective = more reliable sentiment)
                weighted_sentiment = polarity * subjectivity
                sentiment_scores.append(weighted_sentiment)
                headlines.append(article['title'])

            avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
            return avg_sentiment, headlines[:5]
            
        except Exception as e:
            logger.error(f"Error fetching news sentiment: {e}")
            return 0.0, []
    
    def get_social_sentiment(self, instrument: str) -> float:
        """Get sentiment from social media (Twitter/Reddit)."""
        if not CONFIG.get("enable_social_sentiment"):
            return 0.0
        
        twitter_sentiment = self.get_twitter_sentiment(instrument)
        reddit_sentiment = self.get_reddit_sentiment(instrument)
        
        # Average social sentiment
        return (twitter_sentiment + reddit_sentiment) / 2.0
    
    def get_twitter_sentiment(self, instrument: str) -> float:
        """Get Twitter sentiment (requires Twitter API v2)."""
        bearer_token = CONFIG.get("twitter_bearer_token")
        if not bearer_token or bearer_token == "YOUR_TWITTER_BEARER_TOKEN":
            return 0.0
        
        try:
            query = instrument.replace("_", "") + " forex OR " + instrument.replace("_", "/")
            url = "https://api.twitter.com/2/tweets/search/recent"
            headers = {"Authorization": f"Bearer {bearer_token}"}
            params = {
                "query": query,
                "max_results": 50,
                "tweet.fields": "created_at,lang"
            }
            
            resp = requests.get(url, headers=headers, params=params, timeout=15)
            resp.raise_for_status()
            
            tweets = resp.json().get("data", [])
            if not tweets:
                return 0.0
            
            sentiments = []
            for tweet in tweets:
                if tweet.get("lang") != "en":
                    continue
                text = tweet.get("text", "")
                sentiment = TextBlob(text).sentiment.polarity
                sentiments.append(sentiment)
            
            return np.mean(sentiments) if sentiments else 0.0
            
        except Exception as e:
            logger.debug(f"Twitter sentiment error: {e}")
            return 0.0
    
    def get_reddit_sentiment(self, instrument: str) -> float:
        """Get Reddit sentiment from forex-related subreddits."""
        client_id = CONFIG.get("reddit_client_id")
        client_secret = CONFIG.get("reddit_client_secret")
        
        if not client_id or client_id == "YOUR_REDDIT_CLIENT_ID":
            return 0.0
        
        try:
            # Get Reddit OAuth token
            auth = requests.auth.HTTPBasicAuth(client_id, client_secret)
            data = {
                "grant_type": "client_credentials",
                "username": "forex_bot",
                "password": "dummy"
            }
            headers = {"User-Agent": "ForexBot/1.0"}
            
            resp = requests.post("https://www.reddit.com/api/v1/access_token",
                               auth=auth, data=data, headers=headers, timeout=10)
            token = resp.json().get("access_token")
            
            if not token:
                return 0.0
            
            # Search relevant subreddits
            headers = {
                "Authorization": f"Bearer {token}",
                "User-Agent": "ForexBot/1.0"
            }
            
            query = instrument.replace("_", "")
            url = f"https://oauth.reddit.com/r/Forex+wallstreetbets/search"
            params = {
                "q": query,
                "limit": 50,
                "sort": "new",
                "t": "day"
            }
            
            resp = requests.get(url, headers=headers, params=params, timeout=15)
            posts = resp.json().get("data", {}).get("children", [])
            
            sentiments = []
            for post in posts:
                title = post.get("data", {}).get("title", "")
                selftext = post.get("data", {}).get("selftext", "")
                text = title + " " + selftext
                
                sentiment = TextBlob(text).sentiment.polarity
                sentiments.append(sentiment)
            
            return np.mean(sentiments) if sentiments else 0.0
            
        except Exception as e:
            logger.debug(f"Reddit sentiment error: {e}")
            return 0.0
    
    def store_sentiment(self, instrument: str, sentiment_breakdown: Dict[str, float]):
        """Store sentiment data in database."""
        try:
            conn = sqlite3.connect('trades.db')
            c = conn.cursor()
            c.execute('''
                INSERT INTO market_sentiment 
                (timestamp, instrument, news_sentiment, social_sentiment, combined_sentiment, sentiment_sources)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (datetime.now(timezone.utc).isoformat(), instrument,
                  sentiment_breakdown["news"], sentiment_breakdown["social"],
                  sentiment_breakdown["combined"], json.dumps(sentiment_breakdown)))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error storing sentiment: {e}")

sentiment_analyzer = EnhancedSentimentAnalyzer()

# -------------------------
# NEW: Correlation Analysis
# -------------------------
def calculate_correlation_matrix() -> Dict[str, Dict[str, float]]:
    """Calculate correlation between instruments."""
    if not CONFIG.get("enable_correlation_analysis"):
        return {}
    
    try:
        instruments = CONFIG.get("correlation_instruments", [INSTRUMENT])
        correlation_data = {}
        
        # Fetch recent candles for all instruments
        instrument_prices = {}
        for inst in instruments:
            try:
                df = get_candles(inst, 100, GRANULARITY)
                instrument_prices[inst] = df['close'].values
            except:
                continue
        
        if len(instrument_prices) < 2:
            return {}
        
        # Calculate correlations
        correlation_matrix = {}
        instruments_list = list(instrument_prices.keys())
        
        for i, inst1 in enumerate(instruments_list):
            correlation_matrix[inst1] = {}
            for inst2 in instruments_list:
                if inst1 == inst2:
                    correlation_matrix[inst1][inst2] = 1.0
                else:
                    # Ensure same length
                    min_len = min(len(instrument_prices[inst1]), len(instrument_prices[inst2]))
                    prices1 = instrument_prices[inst1][-min_len:]
                    prices2 = instrument_prices[inst2][-min_len:]
                    
                    corr = np.corrcoef(prices1, prices2)[0, 1]
                    correlation_matrix[inst1][inst2] = corr
        
        # Store in database
        store_correlations(correlation_matrix)
        
        return correlation_matrix
        
    except Exception as e:
        logger.error(f"Error calculating correlations: {e}")
        return {}

def store_correlations(correlation_matrix: Dict[str, Dict[str, float]]):
    """Store correlation data."""
    try:
        conn = sqlite3.connect('trades.db')
        c = conn.cursor()
        
        for inst1, correlations in correlation_matrix.items():
            for inst2, corr_value in correlations.items():
                if inst1 != inst2:
                    pair = f"{inst1}_{inst2}"
                    c.execute('''
                        INSERT INTO correlations 
                        (timestamp, instrument_pair, correlation_value, timeframe)
                        VALUES (?, ?, ?, ?)
                    ''', (datetime.now(timezone.utc).isoformat(), pair, corr_value, GRANULARITY))
        
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error storing correlations: {e}")

def check_correlation_exposure() -> bool:
    """Check if current positions have high correlation exposure."""
    if not CONFIG.get("enable_correlation_analysis"):
        return True
    
    try:
        correlation_matrix = calculate_correlation_matrix()
        if not correlation_matrix:
            return True
        
        # Check correlation with open positions
        open_positions = get_open_positions()
        if not open_positions:
            return True
        
        for position in open_positions:
            pos_instrument = position['instrument']
            if pos_instrument in correlation_matrix.get(INSTRUMENT, {}):
                corr = abs(correlation_matrix[INSTRUMENT][pos_instrument])
                threshold = CONFIG.get("correlation_threshold", 0.7)
                
                if corr > threshold:
                    logger.warning(f"High correlation detected: {INSTRUMENT} & {pos_instrument} = {corr:.2f}")
                    return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error checking correlation: {e}")
        return True

# -------------------------
# NEW: Automated Withdrawal Preparation
# -------------------------
def prepare_withdrawal(balance: float, daily_pnl: float):
    """Prepare withdrawal instructions (MANUAL EXECUTION REQUIRED)."""
    if not CONFIG.get("enable_auto_withdrawal_prep"):
        return
    
    min_profit = CONFIG.get("min_profit_for_withdrawal", 100.0)
    
    if daily_pnl < min_profit:
        return
    
    withdrawal_pct = CONFIG.get("withdrawal_percentage", 0.80)
    withdrawal_amount = daily_pnl * withdrawal_pct
    
    # Generate detailed withdrawal instructions
    instructions = generate_withdrawal_instructions(withdrawal_amount)
    
    # Store withdrawal record
    store_withdrawal_request(withdrawal_amount, instructions, STATE)
    
    # Send comprehensive alert
    send_withdrawal_alert(withdrawal_amount, instructions)
    
    logger.info(f"💰 WITHDRAWAL PREPARED: ${withdrawal_amount:.2f}")
    logger.info("=" * 60)
    logger.info(instructions)
    logger.info("=" * 60)

def generate_withdrawal_instructions(amount: float) -> str:
    """Generate detailed step-by-step withdrawal instructions."""
    mpesa_phone = CONFIG.get("mpesa_phone", "")
    bank_account = CONFIG.get("bank_account", "")
    bank_name = CONFIG.get("bank_name", "")
    
    instructions = f"""
╔══════════════════════════════════════════════════════════╗
║          WITHDRAWAL INSTRUCTION - ACTION REQUIRED         ║
╚══════════════════════════════════════════════════════════╝

📊 PROFIT WITHDRAWAL READY
   Amount to Withdraw: ${amount:.2f}
   Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}

⚠️  IMPORTANT: OANDA does not support automated withdrawals.
    You must manually process this withdrawal.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔐 STEP-BY-STEP WITHDRAWAL PROCESS:

1. LOG INTO OANDA ACCOUNT
   • Go to: https://www.oanda.com
   • Login with your credentials
   • Navigate to "Funds Management" or "Withdraw"

2. INITIATE WITHDRAWAL
   • Select withdrawal amount: ${amount:.2f}
   • Choose withdrawal method:
     
     Option A - M-PESA (Kenya):
     ✓ Recipient Phone: {mpesa_phone if mpesa_phone else '[NOT CONFIGURED]'}
     ✓ Processing Time: 1-2 business days
     ✓ Note: Verify phone number is registered with M-Pesa
     
     Option B - Bank Transfer:
     ✓ Bank Name: {bank_name if bank_name else '[NOT CONFIGURED]'}
     ✓ Account Number: {bank_account if bank_account else '[NOT CONFIGURED]'}
     ✓ Processing Time: 3-5 business days
     ✓ Note: Ensure account details match your OANDA profile

3. CONFIRM WITHDRAWAL
   • Review all details carefully
   • Confirm the transaction
   • Save confirmation number/receipt

4. VERIFY TRANSACTION
   • Check your email for OANDA confirmation
   • Expected arrival time:
     - M-Pesa: 1-2 business days
     - Bank: 3-5 business days

5. UPDATE BOT STATUS (OPTIONAL)
   • Once funds received, mark withdrawal as complete
   • This helps track your withdrawal history

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 WITHDRAWAL CHECKLIST:
   □ Logged into OANDA account
   □ Verified withdrawal amount
   □ Selected correct withdrawal method
   □ Confirmed recipient details
   □ Submitted withdrawal request
   □ Saved confirmation number
   □ Checked email confirmation
   □ Funds received and verified

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚡ QUICK LINKS:
   • OANDA Withdrawal: https://www.oanda.com/account/funds/withdraw
   • OANDA Support: https://www.oanda.com/support
   • M-Pesa Support: 234 (from Safaricom line)

📞 NEED HELP?
   • OANDA Support: +44 20 7772 8400
   • Email: support@oanda.com

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️  SECURITY REMINDERS:
   ✓ Never share your OANDA login credentials
   ✓ Verify you're on official OANDA website
   ✓ Double-check M-Pesa/bank account details
   ✓ Keep all confirmation emails/numbers
   ✓ Report any suspicious activity immediately

╔══════════════════════════════════════════════════════════╗
║  This withdrawal must be processed manually through      ║
║  OANDA's official platform for security purposes.        ║
╚══════════════════════════════════════════════════════════╝
"""
    return instructions

def store_withdrawal_request(amount: float, instructions: str, state: 'BotState'):
    """Store withdrawal request in database."""
    try:
        conn = sqlite3.connect('trades.db')
        c = conn.cursor()
        
        withdrawal_method = "M-Pesa" if CONFIG.get("mpesa_phone") else "Bank Transfer"
        
        c.execute('''
            INSERT INTO withdrawals 
            (timestamp, amount, status, withdrawal_method, transaction_details)
            VALUES (?, ?, ?, ?, ?)
        ''', (datetime.now(timezone.utc).isoformat(), amount, "PENDING", 
              withdrawal_method, instructions))
        
        conn.commit()
        conn.close()
        
        state.pending_withdrawals.append({
            'amount': amount,
            'timestamp': datetime.now(timezone.utc),
            'status': 'PENDING'
        })
        
    except Exception as e:
        logger.error(f"Error storing withdrawal request: {e}")

def send_withdrawal_alert(amount: float, instructions: str):
    """Send withdrawal alert through all channels."""
    subject = f"💰 WITHDRAWAL READY: ${amount:.2f}"
    
    send_alert(subject, instructions, telegram_bot)
    
    # Send detailed email if enabled
    if CONFIG.get("enable_email_alerts"):
        send_email_alert(subject, instructions)
    
    # Send Telegram alert
    if telegram_bot and telegram_bot.enabled:
        telegram_summary = f"""
💰 <b>PROFIT WITHDRAWAL READY</b>

Amount: <b>${amount:.2f}</b>
Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}

⚠️ <b>ACTION REQUIRED:</b>
Log into your OANDA account to manually process this withdrawal.

Detailed instructions have been sent via email and logged to the bot.

Quick Link: https://www.oanda.com/account/funds/withdraw
"""
        telegram_bot.send_message(telegram_summary)

# -------------------------
# Indicators (keeping existing code, adding volume analysis)
# -------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower

def stochastic(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series]:
    low_min = df['low'].rolling(window=period).min()
    high_max = df['high'].rolling(window=period).max()
    k = 100 * ((df['close'] - low_min) / (high_max - low_min).replace(0, 1e-10))
    d = k.rolling(window=3).mean()
    return k, d

# NEW: Volume Analysis
def analyze_volume(df: pd.DataFrame) -> Dict[str, float]:
    """Analyze volume patterns for additional signal confirmation."""
    if not CONFIG.get("enable_volume_analysis") or df['volume'].sum() == 0:
        return {"volume_trend": 0, "volume_strength": 0}
    
    try:
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Calculate volume trend
        recent_volume = df['volume'].tail(5).mean()
        avg_volume = df['volume_ma'].iloc[-1]
        volume_trend = (recent_volume - avg_volume) / avg_volume if avg_volume > 0 else 0
        
        # Volume strength indicator
        volume_strength = df['volume_ratio'].iloc[-1]
        
        return {
            "volume_trend": volume_trend,
            "volume_strength": volume_strength,
            "above_average": volume_strength > 1.2
        }
    except:
        return {"volume_trend": 0, "volume_strength": 0}

# Continuing with the rest of the code...:
        logger.error(f"Telegram error: {e}")
    
    def send_photo(self, photo_path: str, caption: str = ""):
        if not self.enabled:
            return
        try:
            url = f"{self.base_url}/sendPhoto"
            with open(photo_path, 'rb') as photo:
                files = {'photo': photo}
                data = {'chat_id': self.chat_id, 'caption': caption}
                resp = requests.post(url, files=files, data=data, timeout=30)
                if resp.status_code == 200:
                    logger.info("Telegram photo sent")
        except Exception as e:
            logger.error(f"Telegram photo error: {e}")

telegram_bot = None  # Will be initialized in main()

# -------------------------
# Alert system
# -------------------------
def send_email_alert(subject: str, message: str):
    if not CONFIG.get("enable_email_alerts") or not CONFIG.get("alert_email"):
        return
    try:
        msg = MIMEMultipart()
        msg['From'] = CONFIG["smtp_username"]
        msg['To'] = CONFIG["alert_email"]
        msg['Subject'] = f"[Trading Bot] {subject}"
        msg.attach(MIMEText(message, 'plain'))
        
        server = smtplib.SMTP(CONFIG["smtp_server"], CONFIG["smtp_port"])
        server.starttls()
        server.login(CONFIG["smtp_username"], CONFIG["smtp_password"])
        server.send_message(msg)
        server.quit()
        logger.info(f"Email alert sent: {subject}")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")

def send_alert(subject: str, message: str, bot: Optional[TelegramBot]):
    send_email_alert(subject, message)
    if bot and bot.enabled:
        bot.send_message(f"<b>{subject}</b>\n\n{message}")

# -------------------------
# State tracking (ENHANCED)
# -------------------------
class BotState:
    def __init__(self):
        self.starting_balance = None
        self.daily_pnl = 0.0
        self.last_reset_date = datetime.now(timezone.utc).date()
        self.position_open = False
        self.last_signal = None
        self.consecutive_losses = 0
        self.current_trade_id = None
        self.current_trade_entry_time = None
        self.current_trade_entry_price = None
        self.current_trade_direction = None
        self.total_trades_today = 0
        self.last_trailing_update = None
        self.ai_predictions = []
        
        # NEW: Enhanced state tracking
        self.pending_withdrawals = []
        self.correlation_matrix = {}
        self.market_sentiment_cache = {}
        self.last_market_analysis = None
        self.volatility_regime = "NORMAL"  # LOW, NORMAL, HIGH
        self.risk_adjustment_factor = 1.0
        
    def reset_daily(self):
        today = datetime.now(timezone.utc).date()
        if today != self.last_reset_date:
            logger.info(f"New trading day. Daily P&L: ${self.daily_pnl:.2f}")
            self.save_daily_performance()
            self.daily_pnl = 0.0
            self.last_reset_date = today
            self.consecutive_losses = 0
            self.total_trades_today = 0
    
    def save_daily_performance(self):
        if self.starting_balance is None:
            return
        try:
            conn = sqlite3.connect('trades.db')
            c = conn.cursor()
            c.execute('''
                SELECT COUNT(*), 
                       SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*), 0)
                FROM trades 
                WHERE DATE(timestamp) = ? AND status = 'CLOSED' AND pnl IS NOT NULL
            ''', (self.last_reset_date.isoformat(),))
            
            result = c.fetchone()
            trades_count = result[0] or 0
            win_rate = result[1] or 0.0
            ending_balance = self.starting_balance + self.daily_pnl
            
            # Calculate Sharpe ratio (simplified)
            sharpe_ratio = self.calculate_sharpe_ratio()
            max_drawdown = self.calculate_max_drawdown()
            
            c.execute('''
                INSERT INTO performance 
                (date, starting_balance, ending_balance, daily_pnl, trades_count, win_rate, sharpe_ratio, max_drawdown)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (self.last_reset_date.isoformat(), self.starting_balance, 
                  ending_balance, self.daily_pnl, trades_count, win_rate, sharpe_ratio, max_drawdown))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error saving daily performance: {e}")
    
    def calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio for recent trades."""
        try:
            conn = sqlite3.connect('trades.db')
            c = conn.cursor()
            c.execute('''
                SELECT pnl FROM trades 
                WHERE status = 'CLOSED' AND pnl IS NOT NULL
                ORDER BY timestamp DESC LIMIT 30
            ''')
            pnls = [row[0] for row in c.fetchall()]
            conn.close()
            
            if len(pnls) < 2:
                return 0.0
            
            returns = np.array(pnls)
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return == 0:
                return 0.0
            
            # Assuming risk-free rate of 0
            sharpe = mean_return / std_return
            return sharpe
        except:
            return 0.0
    
    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        try:
            conn = sqlite3.connect('trades.db')
            c = conn.cursor()
            c.execute('''
                SELECT pnl FROM trades 
                WHERE status = 'CLOSED' AND pnl IS NOT NULL
                ORDER BY timestamp ASC
            ''')
            pnls = [row[0] for row in c.fetchall()]
            conn.close()
            
            if len(pnls) < 2:
                return 0.0
            
            cumulative = np.cumsum(pnls)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / (running_max + 1e-10)
            max_dd = np.min(drawdown)
            
            return max_dd
        except:
            return 0.0

STATE = BotState()

# -------------------------
# OANDA API wrappers (keeping existing code)
# -------------------------
def make_api_request(url: str, method: str = 'GET', params: Optional[Dict] = None, json_data: Optional[Dict] = None, retries: int = 3, delay: int = 5) -> Dict[str, Any]:
    for attempt in range(retries):
        try:
            resp = requests.request(method, url, headers=HEADERS, params=params, json=json_data, timeout=15)
            if resp.status_code >= 400:
                logger.warning(f"API Error: {resp.status_code} {resp.reason}")
            resp.raise_for_status()
            return resp.json()
        except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as e:
            logger.warning(f"API request failed on attempt {attempt + 1}/{retries}: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
    raise ConnectionError(f"API request failed after {retries} retries.")

def get_candles(instrument: str, count: int = 200, granularity: str = "M5") -> pd.DataFrame:
    try:
        url = f"{OANDA_API_URL}/v3/instruments/{instrument}/candles"
        params = {"count": count, "granularity": granularity, "price": "M"}
        data = make_api_request(url, params=params)
        
        rows = []
        for c in data["candles"]:
            if not c["complete"] and CONFIG["ignore_incomplete_candles"]:
                continue
            o = float(c["mid"]["o"])
            h = float(c["mid"]["h"])
            l = float(c["mid"]["l"])
            close = float(c["mid"]["c"])
            t = datetime.fromisoformat(c["time"].replace("Z", "+00:00"))
            vol = int(c.get("volume", 0))
            rows.append({"time": t, "open": o, "high": h, "low": l, "close": close, "volume": vol})
        
        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError("No candle data returned")
        return df
    except Exception as e:
        logger.error(f"Error fetching candles: {e}")
        raise

def get_account_summary() -> Dict[str, Any]:
    try:
        url = f"{OANDA_API_URL}/v3/accounts/{ACCOUNT_ID}/summary"
        data = make_api_request(url)
        return data["account"]
    except Exception as e:
        logger.error(f"Error fetching account: {e}")
        raise

def get_open_positions() -> List[Dict[str, Any]]:
    try:
        url = f"{OANDA_API_URL}/v3/accounts/{ACCOUNT_ID}/openPositions"
        data = make_api_request(url)
        return data.get("positions", [])
    except Exception as e:
        logger.error(f"Error fetching positions: {e}")
        raise

def get_open_trades() -> List[Dict[str, Any]]:
    try:
        url = f"{OANDA_API_URL}/v3/accounts/{ACCOUNT_ID}/openTrades"
        data = make_api_request(url)
        return data.get("trades", [])
    except Exception as e:
        logger.error(f"Error fetching trades: {e}")
        return []

def create_order(instrument: str, units: int, take_profit_price: Optional[float], 
                 stop_loss_price: Optional[float]) -> Dict[str, Any]:
    try:
        url = f"{OANDA_API_URL}/v3/accounts/{ACCOUNT_ID}/orders"
        order = {
            "order": {
                "instrument": instrument,
                "units": str(units),
                "type": "MARKET",
                "positionFill": "DEFAULT"
            }
        }
        if take_profit_price is not None:
            order["order"]["takeProfitOnFill"] = {"price": f"{take_profit_price:.5f}"}
        if stop_loss_price is not None:
            order["order"]["stopLossOnFill"] = {"price": f"{stop_loss_price:.5f}"}
        
        logger.info(f"Placing order: {units} units @ TP={take_profit_price:.5f}, SL={stop_loss_price:.5f}")
        return make_api_request(url, method='POST', json_data=order)
    except Exception as e:
        logger.error(f"Error creating order: {e}")
        raise

def close_position(instrument: str) -> Dict[str, Any]:
    try:
        url = f"{OANDA_API_URL}/v3/accounts/{ACCOUNT_ID}/positions/{instrument}/close"
        payload = {"longUnits": "ALL", "shortUnits": "ALL"}
        data = make_api_request(url, method='PUT', json_data=payload)
        logger.info(f"Closed position for {instrument}")
        return data
    except Exception as e:
        logger.error(f"Error closing position: {e}")
        raise

def get_current_price(instrument: str) -> float:
    try:
        url = f"{OANDA_API_URL}/v3/accounts/{ACCOUNT_ID}/pricing"
        params = {"instruments": instrument}
        data = make_api_request(url, params=params)
        prices = data["prices"]
        if not prices:
            raise ValueError("No price data")
        bid = float(prices[0]["bids"][0]["price"])
        ask = float(prices[0]["asks"][0]["price"])
        return (bid + ask) / 2.0
    except Exception as e:
        logger.error(f"Error fetching current price: {e}")
        return 0.0      
    