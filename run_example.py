import os
import warnings
import numpy as np
import pandas as pd
import ta
from sklearn.preprocessing import MinMaxScaler
from tensortrade.data.cdd import CryptoDataDownload
from tensortrade.env.default import create as create_env
from tensortrade.agents import DQNAgent
from tensortrade.feed.core import DataFeed, Stream
from tensortrade.feed.core.base import NameSpace
from tensortrade.env.default.actions import BSH
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.instruments import USD, BTC
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.env.default.rewards import PBR

# Settings
n_steps = 500
n_episodes = 2
window_size = 30
memory_capacity = n_steps * 10
save_path = 'agents_test/'

warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
pd.set_option('future.no_silent_downcasting', True)

def fetch_data():
    print("Loading custom CSV data...")
    csv_path = 'nse_data.csv'

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find {csv_path} in the folder. Please move your CSV there and rename it to nse_data.csv")

    df = pd.read_csv(csv_path)

    # Filter for RELIANCE only
    if 'Symbol' in df.columns:
        df = df[df['Symbol'] == 'RELIANCE'].copy()
    elif 'symbol' in df.columns:
        df = df[df['symbol'] == 'RELIANCE'].copy()

    # --- SMART COLUMN MATCHER ---
    # This tries to find the right columns even if they are named 'Close Price' or 'Date'
    mapping = {}
    possible_names = {
        'date': ['date', 'Date', 'timestamp', 'Timestamp', 'Time'],
        'open': ['open', 'Open', 'Open Price'],
        'high': ['high', 'High', 'High Price'],
        'low': ['low', 'Low', 'Low Price'],
        'close': ['close', 'Close', 'Close Price'],
        'volume': ['volume', 'Volume', 'Vol']
    }

    for target, options in possible_names.items():
        for opt in options:
            if opt in df.columns:
                mapping[opt] = target
                break

    if len(mapping) < 6:
        print(f"Found columns: {list(df.columns)}")
        raise ValueError(f"Could not find all required columns. Found: {list(mapping.values())}")

    df = df.rename(columns=mapping)
    df = df[['date', 'open', 'high', 'low', 'close', 'volume']]

    # Cleaning
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype(np.float64) / 1e6 # Scale to millions to avoid overflow
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

def rsi(price, period):
    r = price.diff()
    upside = r.clip(lower=0)
    downside = -r.clip(upper=0)
    rs = upside.ewm(alpha=1 / period).mean() / downside.ewm(alpha=1 / period).mean()
    return 100*(1 - (1 + rs) ** -1)

def macd(price, fast, slow, signal):
    fm = price.ewm(span=fast, adjust=False).mean()
    sm = price.ewm(span=slow, adjust=False).mean()
    md = fm - sm
    return md - md.ewm(span=signal, adjust=False).mean()

def generate_features(data):
    print("Generating features...")
    df = data.copy()
    ta.add_all_ta_features(df, 'open', 'high', 'low', 'close', 'volume', fillna=True)

    df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
    df = df.set_index('date')

    features = pd.DataFrame.from_dict({
        'prev_close': df['Close'].shift(1),
        'rsi_14': rsi(df['Close'], period=14),
        'macd_normal': macd(df['Close'], 12, 26, 9),
    })
    data = pd.concat([df, features], axis='columns').ffill().dropna()
    data = data.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
    return data.reset_index()

# PBR used instead of custom scheme

def main():
    try:
        data = fetch_data()
        data = generate_features(data)

        X_train = data.iloc[:10000].copy()

        cols = [c for c in X_train.columns if c != 'date']
        X_scaled = X_train[cols].copy()
        for c in cols:
            scaler = MinMaxScaler()
            X_scaled[c] = scaler.fit_transform(X_train[[c]]).flatten()

        # Using NSE-like naming for the instruments
        price = Stream.source(list(X_train["close"]), dtype="float").rename("close")
        bitstamp = Exchange("nse", service=execute_order)(price)
        cash = Wallet(bitstamp, 50000 * USD) # Generic currency
        asset = Wallet(bitstamp, 0 * BTC)    # Generic stock asset
        portfolio = Portfolio(USD, [cash, asset])

        # Reward and Action Scheme
        reward_scheme = PBR(price=price)
        action_scheme = BSH(cash=cash, asset=asset).attach(reward_scheme)

        # Remove close from cols to avoid duplication in feed
        feature_cols = [c for c in cols if c != 'close']
        features = [price] + [Stream.source(list(X_scaled[c]), dtype="float").rename(c) for c in feature_cols]

        feed = DataFeed(features)
        feed.compile()

        env = create_env(
            portfolio=portfolio,
            action_scheme=action_scheme,
            reward_scheme=reward_scheme,
            feed=feed,
            window_size=window_size
        )

        print("Training Agent on NSE Data...")
        agent = DQNAgent(env)

        batch_size = ((n_steps - window_size - 1) // 4)

        agent.train(batch_size=batch_size, n_steps=n_steps, n_episodes=n_episodes,
                    memory_capacity=memory_capacity, save_path=save_path)

        print("\nTraining Complete!")
        print(f"Final Net Worth: {portfolio.net_worth}")

    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    main()
