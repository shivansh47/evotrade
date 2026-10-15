import os
import warnings
import numpy as np
import pandas as pd
import ta
from sklearn.preprocessing import MinMaxScaler
from tensortrade.env.default import create as create_env
from tensortrade.agents import DQNAgent
from tensortrade.feed.core import DataFeed, Stream
from tensortrade.env.default.actions import BSH
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.instruments import USD, BTC
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.env.default.rewards import PBR

# Settings
n_steps = 1000
n_episodes = 1
window_size = 20
save_path = 'agents_test/'

warnings.filterwarnings("ignore")

def load_and_prepare_data(csv_path):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Filter for RELIANCE
    if 'Symbol' in df.columns:
        df = df[df['Symbol'] == 'RELIANCE'].copy()

    # Column Matching
    mapping = {}
    possible = {
        'date': ['date', 'Date', 'timestamp', 'Timestamp'],
        'open': ['open', 'Open'],
        'high': ['high', 'High'],
        'low': ['low', 'Low'],
        'close': ['close', 'Close'],
        'volume': ['volume', 'Volume']
    }
    for target, opts in possible.items():
        for opt in opts:
            if opt in df.columns:
                mapping[opt] = target
                break

    df = df.rename(columns=mapping)
    df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Feature Engineering
    print("Generating features...")
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['macd'] = ta.trend.macd(df['close'])
    df = df.ffill().dropna()

    return df

def main():
    try:
        # 1. Data
        df = load_and_prepare_data('nse_data.csv')

        # Slice data for training
        train_df = df.iloc[:2000].copy()

        # Scaling
        cols_to_scale = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd']
        scaled_data = train_df.copy()
        for col in cols_to_scale:
            scaler = MinMaxScaler()
            scaled_data[col] = scaler.fit_transform(train_df[[col]]).flatten()

        # 2. Streams
        # The price stream MUST match the trading pair for the Observer to find it
        price_stream = Stream.source(train_df['close'].tolist(), dtype="float").rename("USD-BTC")

        # Feature streams
        features = [
            Stream.source(scaled_data[col].tolist(), dtype="float").rename(col)
            for col in cols_to_scale
        ]

        # Feed must include everything the env needs
        # We already have 'close' in features, so just use features
        feed = DataFeed(features)
        feed.compile()

        # 3. OMS Setup
        exchange = Exchange("nse", service=execute_order)(price_stream)
        cash = Wallet(exchange, 10000 * USD)
        asset = Wallet(exchange, 0 * BTC)
        portfolio = Portfolio(USD, [cash, asset])

        # 4. Environment
        reward_scheme = PBR(price=price_stream)
        action_scheme = BSH(cash=cash, asset=asset).attach(reward_scheme)

        env = create_env(
            portfolio=portfolio,
            action_scheme=action_scheme,
            reward_scheme=reward_scheme,
            feed=feed,
            window_size=window_size
        )

        # 5. Agent
        print("Training Agent...")
        agent = DQNAgent(env)
        agent.train(
            batch_size=32,
            n_steps=n_steps,
            n_episodes=n_episodes,
            save_path=save_path
        )

        print(f"\nSuccess! Final Net Worth: {portfolio.net_worth}")

    except Exception as e:
        print(f"FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
