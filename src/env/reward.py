"""
Reward Function for Trading Environment.

Reward = PnL - Transaction Cost

The reward incentivizes profitable trading while penalizing excessive trading
through transaction costs.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RewardConfig:
    """Configuration for reward calculation."""

    # Transaction cost as fraction of trade value (0.1% = 0.001)
    transaction_cost: float = 0.001

    # Risk aversion parameter (penalty for variance)
    risk_aversion: float = 0.0

    # Time decay penalty (per step)
    time_decay: float = 0.0

    # Reward scaling factor
    reward_scale: float = 1.0

    # Position penalty to discourage large positions
    position_penalty_rate: float = 0.0


def compute_reward(
    position: int,
    price_change: float,
    transaction_cost: float,
    config: Optional[RewardConfig] = None,
) -> float:
    """
    Compute step reward.

    Reward = Position * Price_Change - Transaction_Cost

    Args:
        position: Current position after action (-1, 0, or 1)
        price_change: Price change from previous step (current - previous)
        transaction_cost: Transaction cost incurred this step
        config: Reward configuration

    Returns:
        Reward value (scaled)
    """
    if config is None:
        config = RewardConfig()

    # PnL from position
    pnl = position * price_change

    # Total reward = PnL - costs
    reward = pnl - transaction_cost

    # Optional position penalty
    if config.position_penalty_rate > 0:
        reward -= config.position_penalty_rate * abs(position)

    # Scale reward
    reward *= config.reward_scale

    return reward


def compute_pnl(
    position: int,
    entry_price: float,
    current_price: float,
) -> float:
    """
    Compute unrealized PnL for current position.

    Args:
        position: Current position
        entry_price: Average entry price
        current_price: Current market price

    Returns:
        Unrealized PnL
    """
    if position == 0 or entry_price == 0:
        return 0.0

    return position * (current_price - entry_price)


def compute_return_pct(
    position: int,
    entry_price: float,
    current_price: float,
) -> float:
    """
    Compute return percentage for current position.

    Args:
        position: Current position
        entry_price: Average entry price
        current_price: Current market price

    Returns:
        Return as percentage
    """
    if position == 0 or entry_price == 0:
        return 0.0

    return (current_price - entry_price) / entry_price * 100 * (1 if position > 0 else -1)
