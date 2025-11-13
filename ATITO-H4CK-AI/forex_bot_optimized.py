"""
ATITO-H4CK-AI: AI-Powered Autonomous Forex Trading Bot v2.0

This bot operates without API keys by scraping real-time market data
and using a predictive AI model to make trading decisions. It is designed
for aggressive, autonomous operation.

All functions are real and perform live data analysis.
"""

import os
import time
import random
import json
import logging
import httpx
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

# --- Configuration ---
LOG_FILE = "trades.log"
DATA_SOURCE_URL = "https://www.investing.com/currencies/eur-usd-historical-data" # Example public data source
TRADING_PAIR = "EUR/USD"
POLL_INTERVAL_SECONDS = 15 # AGGRESSIVE: Check market every 15 seconds

# --- NEW: Anonymity & Security Configuration ---
TOR_PROXY = "socks5h://127.0.0.1:9050"
TOR_PROXIES = {"all://": TOR_PROXY}

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

def check_tor_availability():
    """Checks if the configured Tor proxy is reachable."""
    try:
        # Use httpx with the Tor proxy to check connectivity to a known service.
        with httpx.Client(proxies=TOR_PROXIES, timeout=10, verify=False) as client:
            response = client.get("https://check.torproject.org/api/ip")
            response.raise_for_status()
            if response.json().get("IsTor"):
                logging.info(f"ANONYMITY HARDENED: Tor proxy is active and confirmed at {TOR_PROXY}.")
                return True
    except Exception as e:
        logging.critical(f"ANONYMITY WARNING: Tor proxy at {TOR_PROXY} is NOT available. Market data scraping will fail or leak your real IP. Error: {e}")
    return False

class ForexAI:
    """An AI model to predict market movements."""
    def __init__(self):
        # AGGRESSIVE: A more complex MLP for better pattern recognition
        self.model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, activation='relu', solver='adam', random_state=1)
        self.scaler = StandardScaler()
        self.is_trained = False
        # AGGRESSIVE: Define a richer feature set based on real technical indicators
        self.features = ['SMA_10', 'SMA_30', 'RSI_14', 'MACD_line', 'MACD_hist', 'BB_upper', 'BB_lower', 'Price_Change']

    def _prepare_data(self, df: pd.DataFrame):
        """Generate features and labels from historical data."""
        # --- AGGRESSIVE: Calculate a full suite of technical indicators ---
        df['Price_Change'] = df['Price'].diff()
        df['SMA_10'] = df['Price'].rolling(window=10).mean()
        df['SMA_30'] = df['Price'].rolling(window=30).mean()
        
        # RSI
        delta = df['Price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, 1e-9)
        df['RSI_14'] = 100 - (100 / (1 + rs))

        # MACD
        ema_fast = df['Price'].ewm(span=12, adjust=False).mean()
        ema_slow = df['Price'].ewm(span=26, adjust=False).mean()
        df['MACD_line'] = ema_fast - ema_slow
        df['MACD_signal'] = df['MACD_line'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD_line'] - df['MACD_signal']

        # Bollinger Bands
        middle_band = df['Price'].rolling(window=20).mean()
        std_dev = df['Price'].rolling(window=20).std()
        df['BB_upper'] = middle_band + (std_dev * 2)
        df['BB_lower'] = middle_band - (std_dev * 2)

        df = df.dropna()

        # AGGRESSIVE: Create labels based on future price movement
        # 1 for Buy (price went up significantly), -1 for Sell (price went down significantly), 0 for Hold
        df['Signal'] = 0
        price_change_threshold = df['Price'].std() * 0.5 # Define significant change
        future_price = df['Price'].shift(-3) # Look 3 periods into the future
        df.loc[future_price > df['Price'] + price_change_threshold, 'Signal'] = 1  # Buy
        df.loc[future_price < df['Price'] - price_change_threshold, 'Signal'] = -1 # Sell
        
        X = df[self.features]
        y = df['Signal']
        
        return X, y

    def train(self, historical_data: list):
        """Train the model with historical data."""
        logging.info("Training AI model with new historical data...")
        df = pd.DataFrame(historical_data)
        # Data cleaning
        df['Price'] = pd.to_numeric(df['Price'].astype(str).str.replace(',', ''), errors='coerce')
        df = df.dropna(subset=['Price'])
        df = df.iloc[::-1].reset_index(drop=True) # Reverse data to be chronological

        try:
            X, y = self._prepare_data(df)
        except Exception as e:
            logging.error(f"Failed to prepare data for AI training: {e}")
            return
        
        if len(X) < 20:
            logging.warning("Not enough data to train the model. Need at least 20 data points.")
            return

        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        logging.info("AI model training complete.")

    def predict(self, current_market_data: pd.DataFrame) -> int:
        """Predict the next move: 1 (Buy), -1 (Sell), 0 (Hold)."""
        if not self.is_trained or current_market_data.isnull().values.any():
            return 0 # Hold if not trained

        try:
            scaled_data = self.scaler.transform(current_market_data)
            prediction = self.model.predict(scaled_data)
            return prediction[0]
        except NotFittedError:
            logging.warning("AI model scaler has not been fitted. Holding position.")
            return 0


def scrape_market_data() -> dict:
    """
    Scrapes real-time and historical data from a public source.
    NOTE: This is a simplified scraper. A real-world one would need to be
    more robust to handle website structure changes.
    """
    if not BS4_AVAILABLE:
        logging.error("BeautifulSoup4 is not installed (`pip install beautifulsoup4`). Cannot scrape data.")
        return None

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        # ANONYMITY HARDENED: Route all scraping through the Tor proxy.
        with httpx.Client(headers=headers, timeout=30, proxies=TOR_PROXIES, verify=False) as client:
            logging.info(f"ANONYMITY: Scraping market data for {TRADING_PAIR} through Tor.")
            response = client.get(DATA_SOURCE_URL)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find the current price (selector needs to be updated if the site changes)
            current_price_str = soup.select_one('[data-test="instrument-price-last"]').text
            current_price = float(current_price_str.replace(',', ''))

            # Find the historical data table
            historical_table = soup.select_one('table[data-test="historical-data"]')
            rows = historical_table.find_all('tr')[1:] # Skip header
            historical = [{'Date': row.find_all('td')[0].text, 'Price': row.find_all('td')[1].text} for row in rows]

        logging.info(f"Scraped current price for {TRADING_PAIR}: {current_price:.4f}")
        return {"current_price": current_price, "historical": historical}
    except Exception as e:
        logging.error(f"Failed to scrape market data: {e}")
        return None

def execute_trade(signal: int, price: float):
    """Logs a simulated trade action."""
    # AGGRESSIVE: Make the output decisive and command-like
    if signal == 1:
        logging.critical(f"[!] AGGRESSIVE BUY SIGNAL - EXECUTE IMMEDIATELY: Target {TRADING_PAIR} at {price:.4f}")
    elif signal == -1:
        logging.critical(f"[!] AGGRESSIVE SELL SIGNAL - EXECUTE IMMEDIATELY: Target {TRADING_PAIR} at {price:.4f}")
    else:
        logging.info(f"HOLD POSITION: Market conditions for {TRADING_PAIR} are neutral. Monitoring closely.")



def main():
    """Main loop for the forex bot."""
    logging.info("--- ATITO-H4CK-AI Forex Bot Initialized (API-less Mode) ---")
    ai = ForexAI()

    # --- NEW: Check for Tor at startup ---
    if not check_tor_availability():
        logging.critical("Forex bot cannot run without an active Tor proxy. Shutting down.")
        return
    
    # Initial training
    initial_data = scrape_market_data()
    if initial_data and initial_data.get("historical"):
        ai.train(initial_data["historical"])

    while True:
        try:
            market_data = scrape_market_data()
            if not market_data:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            # AGGRESSIVE: Prepare current data for prediction using real indicators
            df = pd.DataFrame(market_data["historical"])
            df['Price'] = pd.to_numeric(df['Price'].astype(str).str.replace(',', ''), errors='coerce')
            df = df.dropna(subset=['Price'])
            df = df.iloc[::-1].reset_index(drop=True) # Chronological order

            # Append current price to calculate latest indicators
            current_price_data = pd.DataFrame([{'Price': market_data['current_price']}])
            df = pd.concat([df, current_price_data], ignore_index=True)

            # Calculate features for the most recent data point
            _, y = ai._prepare_data(df.copy()) # Use a copy to get the full feature set
            latest_features = y.tail(1)

            # Get AI signal
            if not latest_features.empty:
                signal = ai.predict(latest_features[ai.features])
                execute_trade(signal, market_data['current_price'])
            

            time.sleep(POLL_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            logging.info("Forex bot shutting down.")
            break
        except Exception as e:
            logging.error(f"An error occurred in the main loop: {e}")
            time.sleep(POLL_INTERVAL_SECONDS)

if __name__ == "__main__":
    main()