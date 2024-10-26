import pandas as pd
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

class TradingEnv(gym.Env):
    """Custom Environment for trading based on historical data."""
    metadata = {'render.modes': ['human']}

    def __init__(self, data, window_size=5):
        super(TradingEnv, self).__init__()
        self.data = data
        self.window_size = window_size

        # Define feature columns
        self.feature_columns = ['close', 'SMA_7', 'SMA_14', 'EMA_7', 'EMA_14', 
                                'MACD', 'RSI', 'Signal', 'Upper Band', 
                                'Lower Band', 'ATR', 'close_percent', 
                                'high_percent', 'low_percent', 'open_percent', 
                                'volume_btc_percent', 'volume_usd_percent', 'sentiment_score_avg']

        self.action_space = gym.spaces.Discrete(3)  # Hold, Buy, Sell
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, 
                                                  shape=(self.window_size * len(self.feature_columns) + 2,),
                                                  dtype=np.float32)

        self.initial_balance = 100.0
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.holdings = 0.0
        self.current_step = 0
        self.total_steps = len(self.data)
        self.done = False
        self.window = np.zeros((self.window_size, len(self.feature_columns)))

        obs = np.concatenate((self.window.flatten(), [self.balance, self.holdings]), axis=None)
        return obs

    def step(self, action):
        current_price = self.data['close'].values[self.current_step]  # Get the current price
        prev_balance = self.balance

        if action == 1 and self.balance >= 100:  # Buy
            shares_to_buy = 100 / current_price
            self.balance -= 100
            self.holdings += shares_to_buy
        elif action == 2 and self.holdings > 0:  # Sell
            sale_value = self.holdings * current_price
            self.balance += sale_value
            self.holdings = 0

        # Move to next step
        self.current_step += 1
        if self.current_step >= self.total_steps:
            self.done = True

        # Update observation window
        self.update_window()

        # Reward is the change in balance
        reward = (self.balance - prev_balance) / self.initial_balance

        # Ensure no NaN in balance or holdings
        self.balance = np.nan_to_num(self.balance)
        self.holdings = np.nan_to_num(self.holdings)

        # Return observation, reward, done, and additional info
        obs = np.concatenate((self.window.flatten(), [self.balance, self.holdings]), axis=None)
        return obs, reward, self.done, {}

    def update_window(self):
        start = max(0, self.current_step - self.window_size + 1)
        end = self.current_step + 1
        self.window = np.pad(self.data[self.feature_columns].values[start:end], 
                             ((self.window_size - (end - start), 0), (0, 0)), 
                             mode='constant')

    def render(self):
        pass

    def close(self):
        pass

# Load the model
model = PPO.load('ppo_trading_agent')

# Load the 2 weeks of data
data = pd.read_csv('2weeks.csv')
data['date'] = pd.to_datetime(data['date'])  # Assuming there is a 'date' column
data.set_index('date', inplace=True)

# Create the trading environment
env = TradingEnv(data)

# Test the trained agent
obs = env.reset()
done = False

# Variables to store performance metrics
balances = []
actions = []
realized_profits = []
total_transaction_fees = 0.0
transaction_count = 0

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)

    # Record performance metrics
    balances.append(env.balance)
    actions.append(action)

    # Update realized profits and transaction fees
    if action == 1:  # Buy
        transaction_count += 1
    elif action == 2:  # Sell
        transaction_count += 1
        profit = (env.balance - 100)  # Adjust according to your initial investment
        realized_profits.append(profit)

# Final calculations
final_balance = env.balance
total_transaction_fees = transaction_count * 0.20  # Assuming 0.20 is the transaction fee
final_realized_profit = sum(realized_profits) if realized_profits else 0

# Calculate Sharpe Ratio
returns = np.array(realized_profits)
mean_return = np.mean(returns)
std_return = np.std(returns)
sharpe_ratio = mean_return / std_return if std_return > 0 else 0  # Avoid division by zero

# Print final results
print("Final Balance:", final_balance)
print("Total Transaction Fees:", total_transaction_fees)
print("Final Realized Profit:", final_realized_profit)
print("Sharpe Ratio:", sharpe_ratio)

# Plot results
plt.figure(figsize=(14, 7))

# Plot Balance Over Time
plt.subplot(2, 1, 1)
plt.plot(balances, label='Balance', color='blue')
plt.title('Balance Over Time')
plt.xlabel('Time Steps')
plt.ylabel('Balance ($)')
plt.legend()

# Plot Realized Profits
plt.subplot(2, 1, 2)
plt.plot(realized_profits, label='Realized Profits', color='green')
plt.title('Realized Profits Over Time')
plt.xlabel('Trade Number')
plt.ylabel('Realized Profit ($)')
plt.legend()

plt.tight_layout()
plt.show()

# Clean up the environment
env.close()
