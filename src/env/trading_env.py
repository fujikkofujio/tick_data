"""
Tick Trading Gymnasium Environment.

Provides a Gymnasium-compatible environment for training RL agents
on tick bar data.

State: 32 dimensions = 28 market features + 4 position information
Action: Discrete 3 values - HOLD(0), BUY(1), SELL(2)
Reward: PnL - Transaction Cost (0.1%)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.actions import Action, ACTION_SPACE_SIZE
from env.reward import RewardConfig, compute_reward


@dataclass
class EnvConfig:
    """Environment configuration."""

    # State dimension (market features)
    state_dim: int = 28

    # Maximum steps per episode
    max_steps: int = 1000

    # Whether to start at random position in data
    random_start: bool = True

    # Maximum position (units)
    max_position: int = 1

    # Initial cash balance
    initial_cash: float = 1_000_000.0

    # Reward configuration
    reward_config: RewardConfig = field(default_factory=RewardConfig)

    # Position info dimensions (position, cash_ratio, unrealized_pnl, steps_in_position)
    position_info_dim: int = 4


class TickTradingEnv(gym.Env):
    """
    Gymnasium environment for tick bar trading.

    Observations include market features and position information.
    Actions are discrete: HOLD, BUY, SELL.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        state_array: np.ndarray,
        price_array: np.ndarray,
        config: Optional[EnvConfig] = None,
    ):
        """
        Initialize trading environment.

        Args:
            state_array: Preprocessed market features (n_samples, n_features)
            price_array: Close prices for each bar (n_samples,)
            config: Environment configuration
        """
        super().__init__()

        self.state_array = state_array.astype(np.float32)
        self.price_array = price_array.astype(np.float32)
        self.config = config or EnvConfig()

        # Validate data
        assert len(state_array) == len(price_array), "State and price arrays must have same length"
        self.n_samples = len(state_array)

        # Total observation dimension
        self.obs_dim = self.config.state_dim + self.config.position_info_dim

        # Define spaces
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        # State variables
        self._position = 0
        self._cash = self.config.initial_cash
        self._entry_price = 0.0
        self._step_idx = 0
        self._start_idx = 0
        self._steps_in_position = 0

        # Episode tracking
        self._episode_trades = []
        self._episode_rewards = []
        self._total_transaction_cost = 0.0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        # Reset state
        self._position = 0
        self._cash = self.config.initial_cash
        self._entry_price = 0.0
        self._steps_in_position = 0

        # Reset tracking
        self._episode_trades = []
        self._episode_rewards = []
        self._total_transaction_cost = 0.0

        # Set starting position
        if self.config.random_start:
            max_start = self.n_samples - self.config.max_steps - 1
            if max_start > 0:
                self._start_idx = self.np_random.integers(0, max_start)
            else:
                self._start_idx = 0
        else:
            self._start_idx = 0

        self._step_idx = 0

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Action to take (0=HOLD, 1=BUY, 2=SELL)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Get current and next price
        current_idx = self._start_idx + self._step_idx
        current_price = self.price_array[current_idx]

        # Store previous position for reward calculation
        prev_position = self._position

        # Execute action
        transaction_cost = self._execute_action(action, current_price)
        self._total_transaction_cost += transaction_cost

        # Move to next step
        self._step_idx += 1
        next_idx = self._start_idx + self._step_idx

        # Calculate price change for reward
        if next_idx < self.n_samples:
            next_price = self.price_array[next_idx]
            price_change = next_price - current_price
        else:
            price_change = 0.0

        # Compute reward
        reward = compute_reward(
            position=prev_position,
            price_change=price_change,
            transaction_cost=transaction_cost,
            config=self.config.reward_config,
        )

        # Apply time decay penalty
        if self.config.reward_config.time_decay > 0:
            reward -= self.config.reward_config.time_decay

        self._episode_rewards.append(reward)

        # Update position time
        if self._position != 0:
            self._steps_in_position += 1
        else:
            self._steps_in_position = 0

        # Check termination
        terminated = False
        truncated = False

        # Episode ends when max steps reached or data exhausted
        if self._step_idx >= self.config.max_steps:
            truncated = True
        if next_idx >= self.n_samples - 1:
            terminated = True

        # Force close position at end
        if terminated or truncated:
            if self._position != 0:
                # Close position
                close_price = self.price_array[min(next_idx, self.n_samples - 1)]
                self._close_position(close_price)

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _execute_action(self, action: int, price: float) -> float:
        """
        Execute trading action.

        Args:
            action: Action to take
            price: Current price

        Returns:
            Transaction cost incurred
        """
        transaction_cost = 0.0

        if action == Action.BUY:
            if self._position < self.config.max_position:
                # Buy
                trade_value = price
                transaction_cost = trade_value * self.config.reward_config.transaction_cost
                self._cash -= trade_value + transaction_cost
                self._position += 1

                # Update entry price (simple average for multiple units)
                if self._position == 1:
                    self._entry_price = price
                else:
                    # Average cost
                    self._entry_price = (self._entry_price + price) / 2

                self._episode_trades.append({
                    "type": "BUY",
                    "price": price,
                    "step": self._step_idx,
                })
                self._steps_in_position = 0

        elif action == Action.SELL:
            if self._position > -self.config.max_position:
                # Sell / short
                trade_value = price
                transaction_cost = trade_value * self.config.reward_config.transaction_cost
                self._cash += trade_value - transaction_cost

                # Record trade result if closing position
                if self._position > 0:
                    pnl = price - self._entry_price
                    self._episode_trades.append({
                        "type": "SELL",
                        "price": price,
                        "step": self._step_idx,
                        "pnl": pnl,
                    })
                else:
                    self._episode_trades.append({
                        "type": "SELL",
                        "price": price,
                        "step": self._step_idx,
                    })
                    if self._position == -1:
                        self._entry_price = price
                    else:
                        self._entry_price = (self._entry_price + price) / 2

                self._position -= 1
                self._steps_in_position = 0

        return transaction_cost

    def _close_position(self, price: float) -> float:
        """Close current position at given price."""
        transaction_cost = 0.0

        if self._position > 0:
            # Close long position
            trade_value = price * self._position
            transaction_cost = trade_value * self.config.reward_config.transaction_cost
            self._cash += trade_value - transaction_cost
            pnl = (price - self._entry_price) * self._position
            self._episode_trades.append({
                "type": "CLOSE_LONG",
                "price": price,
                "step": self._step_idx,
                "pnl": pnl,
            })
            self._position = 0

        elif self._position < 0:
            # Close short position
            trade_value = price * abs(self._position)
            transaction_cost = trade_value * self.config.reward_config.transaction_cost
            self._cash -= trade_value + transaction_cost
            pnl = (self._entry_price - price) * abs(self._position)
            self._episode_trades.append({
                "type": "CLOSE_SHORT",
                "price": price,
                "step": self._step_idx,
                "pnl": pnl,
            })
            self._position = 0

        self._entry_price = 0.0
        self._total_transaction_cost += transaction_cost
        return transaction_cost

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        current_idx = self._start_idx + self._step_idx
        current_idx = min(current_idx, self.n_samples - 1)

        # Market features
        market_features = self.state_array[current_idx]

        # Position information (normalized)
        current_price = self.price_array[current_idx]

        position_normalized = self._position / max(self.config.max_position, 1)
        cash_ratio = self._cash / self.config.initial_cash - 1.0  # Centered at 0

        if self._position != 0 and self._entry_price > 0:
            unrealized_pnl = (current_price - self._entry_price) / self._entry_price
            if self._position < 0:
                unrealized_pnl = -unrealized_pnl
        else:
            unrealized_pnl = 0.0

        steps_normalized = self._steps_in_position / 100.0  # Normalize step count

        position_info = np.array([
            position_normalized,
            cash_ratio,
            unrealized_pnl,
            steps_normalized,
        ], dtype=np.float32)

        # Combine features
        obs = np.concatenate([market_features, position_info])

        return obs

    def _get_info(self) -> Dict[str, Any]:
        """Get current info dict."""
        # Calculate statistics
        num_trades = len(self._episode_trades)
        trades_with_pnl = [t for t in self._episode_trades if "pnl" in t]
        total_pnl = sum(t["pnl"] for t in trades_with_pnl) if trades_with_pnl else 0.0
        winning_trades = [t for t in trades_with_pnl if t["pnl"] > 0]
        win_rate = len(winning_trades) / len(trades_with_pnl) if trades_with_pnl else 0.0

        return {
            "step": self._step_idx,
            "position": self._position,
            "cash": self._cash,
            "entry_price": self._entry_price,
            "stats": {
                "num_trades": num_trades,
                "total_pnl": total_pnl,
                "win_rate": win_rate,
                "total_transaction_cost": self._total_transaction_cost,
                "total_reward": sum(self._episode_rewards),
            },
        }

    def render(self) -> None:
        """Render environment state."""
        info = self._get_info()
        current_idx = self._start_idx + self._step_idx
        current_price = self.price_array[min(current_idx, self.n_samples - 1)]

        print(f"Step: {self._step_idx}/{self.config.max_steps}")
        print(f"Position: {self._position}, Entry: {self._entry_price:.2f}, Current: {current_price:.2f}")
        print(f"Cash: {self._cash:.2f}, Trades: {info['stats']['num_trades']}")

    def close(self) -> None:
        """Clean up resources."""
        pass
