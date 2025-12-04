"""
Action Space Definition.

Discrete action space for trading:
- HOLD (0): Do nothing, maintain current position
- BUY (1): Buy/increase position
- SELL (2): Sell/decrease position
"""

from enum import IntEnum
from typing import Tuple


class Action(IntEnum):
    """Trading action enumeration."""
    HOLD = 0
    BUY = 1
    SELL = 2


# Action space size for Gymnasium
ACTION_SPACE_SIZE = 3


def get_action_name(action: int) -> str:
    """Get human-readable action name."""
    return Action(action).name


def apply_action(
    action: int,
    current_position: int,
    current_cash: float,
    price: float,
    max_position: int = 1,
    transaction_cost_rate: float = 0.001,
) -> Tuple[int, float, float, float]:
    """
    Apply trading action and return new state.

    Args:
        action: Action to take (0=HOLD, 1=BUY, 2=SELL)
        current_position: Current position (-max to +max)
        current_cash: Current cash balance
        price: Current price
        max_position: Maximum position size (default 1 for single unit)
        transaction_cost_rate: Transaction cost as fraction of trade value

    Returns:
        Tuple of (new_position, new_cash, trade_value, transaction_cost)
    """
    new_position = current_position
    new_cash = current_cash
    trade_value = 0.0
    transaction_cost = 0.0

    if action == Action.BUY:
        # Buy if not at max long position
        if current_position < max_position:
            trade_value = price
            transaction_cost = trade_value * transaction_cost_rate
            new_cash = current_cash - trade_value - transaction_cost
            new_position = current_position + 1

    elif action == Action.SELL:
        # Sell if not at max short position
        if current_position > -max_position:
            trade_value = price
            transaction_cost = trade_value * transaction_cost_rate
            new_cash = current_cash + trade_value - transaction_cost
            new_position = current_position - 1

    # HOLD: no change

    return new_position, new_cash, trade_value, transaction_cost
