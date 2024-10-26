import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import pandas as pd
import time

class TradingEnv(gym.Env):
    """Custom Environment for stock trading with new indicators."""
    metadata = {'render.modes': ['human']}

    def __init__(self, window_size=5):
        super(TradingEnv, self).__init__()
        self.window_size = window_size  # Observation window size
        
        # New feature columns except 'close'
        self.feature_columns = ['close', 'SMA_7', 'SMA_14', 'EMA_7', 'EMA_14', 'MACD', 'RSI', 
                                'Signal', 'Upper Band', 'Lower Band', 'ATR', 'close_percent', 
                                'high_percent', 'low_percent', 'open_percent', 
                                'volume_btc_percent', 'volume_usd_percent', 'sentiment_score_avg'] 
        self.num_features = len(self.feature_columns)
        
        # Define action and observation space
        self.action_space = spaces.Discrete(3)

        # Adjust observation space to accommodate features instead of just price
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, 
                                            shape=(self.window_size * self.num_features + 2,),
                                            dtype=np.float32)

        self.initial_balance = 100.0
        self.realized_profit = 0.0
        self.transaction_fee_rate = 0.20
        self.buy_prices = []
        self.sell_prices = []
        self.total_transaction_fees = 0.0  # Track total transaction fees
        self.reset()

    def reset(self, seed=None, options=None):
        # Load merged data and randomly select two weeks
        merged_data = pd.read_csv('merged_data.csv')
        merged_data['date'] = pd.to_datetime(merged_data['date']) 
        merged_data.set_index('date', inplace=True)

        # Select a random date for the start of the two-week window
        start_date = merged_data.index.min() + pd.Timedelta(days=np.random.randint(0, (merged_data.index.max() - merged_data.index.min()).days - 14))
        end_date = start_date + pd.Timedelta(days=14)

        # Filter data for the selected date range
        self.data = merged_data.loc[start_date:end_date]
        self.prices = self.data['close'].values  # Close prices for plotting only
        self.indicators = self.data[self.feature_columns].values  # Only use the selected indicators

        self.balance = self.initial_balance
        self.holdings = 0.0
        self.current_step = 0
        self.total_steps = len(self.prices)
        self.realized_profit = 0.0
        self.done = False
        self.window = np.zeros((self.window_size, self.num_features))  # Observation window with all indicators
        self.update_window()  # Initialize window
        obs = np.concatenate((self.window.flatten(), [self.balance, self.holdings]), axis=None)
        return obs, {}

    def update_window(self):
        start = max(0, self.current_step - self.window_size + 1)
        end = self.current_step + 1
        self.window = np.pad(self.indicators[start:end], 
                             ((self.window_size - (end - start), 0), (0, 0)), 
                             mode='constant')

    def step(self, action):
        current_price = self.prices[self.current_step]  # Close price for plotting only
        prev_realized_profit = self.realized_profit

        # Action interpretation
        if action == 0:
            pass  # Hold
        elif action == 1 and self.balance >= 100:
            shares_to_buy = 100 / current_price
            self.balance -= 100
            self.total_transaction_fees += self.transaction_fee_rate  # Add transaction fee
            self.holdings += shares_to_buy
            self.buy_prices.append(current_price)
        elif action == 2 and self.holdings > 0:
            sale_value = self.holdings * current_price
            self.balance += sale_value
            self.total_transaction_fees += self.transaction_fee_rate  # Add transaction fee
            profit = self.balance - self.initial_balance
            self.realized_profit = profit
            self.holdings = 0
            self.sell_prices.append(current_price)

        # Move to next step
        self.current_step += 1
        if self.current_step >= self.total_steps:
            self.done = True

        # Update observation window
        self.update_window()

        # Ensure no NaN in balance or holdings
        self.balance = np.nan_to_num(self.balance)
        self.holdings = np.nan_to_num(self.holdings)
        self.realized_profit = np.nan_to_num(self.realized_profit)

        # Calculate reward based on realized profit
        reward = (self.realized_profit - prev_realized_profit) / self.initial_balance
        reward = np.clip(reward, -10, 10)

        # Concatenate updated window with balance and holdings
        obs = np.concatenate((self.window.flatten(), [self.balance, self.holdings]), axis=None)
        
        # Return observation, reward, done, and additional info
        return obs, reward, self.done, False, {}

    def render(self):
        pass

    def close(self):
        pass


start_time = time.time()

# Create and wrap the environment
env = TradingEnv(window_size=5)  # 5-step price window
env = Monitor(env)

# Initialize the PPO agent
model = PPO('MlpPolicy', env, verbose=1, device='cpu', clip_range=0.1, clip_range_vf=0.1, max_grad_norm=0.5, learning_rate=1e-3)

# Train the agent
model.learn(total_timesteps=1000000)  

model.save('ppo_trading_agent')

end_train_time = time.time()

# Test the trained agent
obs, _ = env.reset()
done = False

# Variables to store performance metrics
steps = []
balances = []
holdings = []
realized_profits = []
actions = []
profits = []
prices = []
total_transaction_fees = 0

# Open a file to log the information
with open('trading_log.txt', 'w') as log_file:
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)

        # Record performance metrics
        steps.append(env.current_step)
        balances.append(env.balance)
        holdings.append(env.holdings)
        realized_profits.append(env.realized_profit)
        actions.append(action)
        profit = env.realized_profit
        profits.append(profit)
        prices.append(env.prices[env.current_step - 1])

        # Log the information to the file
        log_message = (f"Action: {['Hold', 'Buy', 'Sell'][action]}, "
                    f"Step: {env.current_step}, "
                    f"Price: {env.prices[env.current_step - 1]:.2f}, "
                    f"Balance: {env.balance:.2f}, "
                    f"Holdings: {env.holdings:.2f}, "
                    f"Realized Profit: {env.realized_profit:.2f},"
                    f"Transaction Fees: {env.total_transaction_fees:.2f}")
        log_file.write(log_message + "\n")
        print(log_message)

# Calculate total transaction fees
transaction_count = 0
for i in range(1, len(actions)):
    if (actions[i-1] == 1 and actions[i] == 2) or \
       (actions[i-1] == 2 and actions[i] == 1) or \
       (actions[i-1] == 0 and actions[i] == 2):
        transaction_count += 1

total_transaction_fees = transaction_count * 0.20

average_buy_price = np.mean(env.buy_prices) if env.buy_prices else 0
average_sell_price = np.mean(env.sell_prices) if env.sell_prices else 0

# Calculate final realized profit and performance metrics
profit = env.realized_profit
print(f"Realized Profit: {profit:.2f}")
print(f"Total Transaction Fees: {total_transaction_fees:.2f}")

# Calculate Sharpe Ratio
returns = np.array(realized_profits)  
mean_return = np.mean(returns)
std_return = np.std(returns)
sharpe_ratio = mean_return / std_return if std_return > 0 else 0  # Avoid division by zero
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")


# Plotting the performance metrics
# First plot: Profit and Price over time
fig, ax1 = plt.subplots(figsize=(18, 10))

# Plot realized profits over time (blue line)
ax1.plot(steps, realized_profits, label='Profit', color='blue', linewidth=2)
ax1.set_xlabel('Steps')
ax1.set_ylabel('Realized Profit', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Create a second y-axis for balance (red line)
ax2 = ax1.twinx()
ax2.plot(steps, balances, label='Balance', color='red', linewidth=2)
ax2.set_ylabel('Balance', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Set titles and show legend
ax1.set_title('Trading Performance')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.show()

# Plot the trading actions over time
plt.figure(figsize=(18, 5))
plt.plot(steps, actions, marker='o', color='purple', linewidth=1)
plt.title('Actions Over Time (0: Hold, 1: Buy, 2: Sell)')
plt.xlabel('Steps')
plt.ylabel('Action')
plt.yticks([0, 1, 2], ['Hold', 'Buy', 'Sell'])
plt.grid()
plt.show()

# Clean up the environment
env.close()
