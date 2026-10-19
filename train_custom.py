import os
import warnings
import numpy as np
import pandas as pd
import ta
from sklearn.preprocessing import StandardScaler
from tensortrade.env.default import create as create_env
from tensortrade.agents import DQNAgent
from tensortrade.feed.core import DataFeed, Stream
from tensortrade.env.default.actions import BSH
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.instruments import Instrument
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.env.default.rewards import PBR

# Settings
n_steps = 1000
n_episodes = 2
window_size = 20
save_path = 'agents_test/'

warnings.filterwarnings("ignore")

def load_and_prepare_data(csv_path, stock_symbol):
    print(f"Loading data for {stock_symbol} from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Filter for specific stock
    if 'Symbol' in df.columns:
        df = df[df['Symbol'] == stock_symbol].copy()
    elif 'symbol' in df.columns:
        df = df[df['symbol'] == stock_symbol].copy()
    else:
        raise ValueError("CSV must contain 'Symbol' or 'symbol' column")

    # --- SMART COLUMN MATCHER ---
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

def main(stock_symbol):
    try:
        # 1. Data
        df = load_and_prepare_data('nse_data.csv', stock_symbol)

        # Slice data for training
        train_df = df.iloc[:2000].copy()

        # Scaling
        cols_to_scale = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd']
        scaled_data = train_df.copy()
        for col in cols_to_scale:
            scaler = StandardScaler()
            scaled_data[col] = scaler.fit_transform(train_df[[col]]).flatten()

        # 2. Streams
        # The price stream MUST match the trading pair format 'Base/Quote'
        price_stream = Stream.source(train_df['close'].tolist(), dtype="float").rename("INR/STOCK")

        # Feature streams
        feature_cols = [c for c in cols_to_scale if c != 'close']
        features = [price_stream] + [
            Stream.source(scaled_data[col].tolist(), dtype="float").rename(col)
            for col in feature_cols
        ]

        # Feed must include everything the env needs
        feed = DataFeed(features)
        feed.compile()

        # 3. OMS Setup
        INR = Instrument("INR", precision=2)
        STOCK = Instrument("STOCK", precision=4)
        initial_cash = 10000
        exchange = Exchange("nse", service=execute_order)(price_stream)
        cash = Wallet(exchange, initial_cash * INR)
        asset = Wallet(exchange, 0 * STOCK)
        portfolio = Portfolio(INR, [cash, asset])

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

        print("\n" + "="*30)
        print("TRADING REPORT")
        print("="*30)
        print(f"Stock:       {stock_symbol}")
        print(f"Steps:       {n_steps}")
        print(f"Episodes:    {n_episodes}")
        print(f"Initial:     {initial_cash} INR")
        print(f"Final Worth: {portfolio.net_worth:.2f} INR")
        print(f"P&L:         {portfolio.net_worth - initial_cash:.2f} INR")
        print("="*30)

    except Exception as e:
        print(f"FATAL ERROR for {stock_symbol}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    stocks = ['RELIANCE', 'TCS', 'HDFCBANK']
    for s in stocks:
        main(s)
