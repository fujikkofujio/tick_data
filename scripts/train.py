"""
Training Script

Run PPO training on tick bar trading environment.
Supports GPU acceleration with 4070Ti Super.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from env.trading_env import TickTradingEnv, EnvConfig
from env.reward import RewardConfig


def check_gpu():
    """Check GPU availability."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"GPU available: {device_name}")
        return "cuda"
    else:
        print("GPU not available, using CPU")
        return "cpu"


def load_data(data_dir: Path):
    """Load preprocessed data."""
    import polars as pl

    state_path = data_dir / "state_array.npy"
    bar_path = data_dir / "processed_features.parquet"

    if not state_path.exists():
        raise FileNotFoundError(f"State array not found: {state_path}")

    state_array = np.load(state_path)
    print(f"Loaded state array: {state_array.shape}")

    # Load price array from bars
    bars = pl.read_parquet(bar_path)
    price_array = bars["close"].to_numpy()
    print(f"Loaded price array: {price_array.shape}")

    # Align lengths (state array may have warmup skipped)
    if len(price_array) > len(state_array):
        offset = len(price_array) - len(state_array)
        price_array = price_array[offset:]

    return state_array, price_array


def make_env_fn(state_array, price_array, config, seed=0):
    """Factory function for creating environments."""
    def _init():
        env = TickTradingEnv(state_array, price_array, config)
        env.reset(seed=seed)
        return Monitor(env)
    return _init


def train(
    data_dir: str = None,
    output_dir: str = None,
    total_timesteps: int = 500_000,
    n_envs: int = 4,
    seed: int = 42,
):
    """
    Train PPO agent on trading environment.

    Args:
        data_dir: Directory with preprocessed data
        output_dir: Directory to save models and logs
        total_timesteps: Total training steps
        n_envs: Number of parallel environments
        seed: Random seed
    """
    # Set paths
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data" / "processed"
    else:
        data_dir = Path(data_dir)

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent.parent / "models" / f"ppo_{timestamp}"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Check GPU
    device = check_gpu()

    # Load data
    state_array, price_array = load_data(data_dir)

    # Environment config
    reward_config = RewardConfig(
        transaction_cost=0.001,  # 0.1% transaction cost
        risk_aversion=0.0,
        time_decay=0.0,
    )

    env_config = EnvConfig(
        state_dim=state_array.shape[1],
        max_steps=1000,
        random_start=True,
        reward_config=reward_config,
    )

    # Create vectorized environment
    print(f"Creating {n_envs} parallel environments...")
    env_fns = [make_env_fn(state_array, price_array, env_config, seed + i) for i in range(n_envs)]

    if n_envs > 1:
        env = SubprocVecEnv(env_fns)
    else:
        env = DummyVecEnv(env_fns)

    # Create evaluation environment
    eval_env = DummyVecEnv([make_env_fn(state_array, price_array, env_config, seed + 100)])

    # PPO hyperparameters (tuned for trading)
    ppo_params = {
        "policy": "MlpPolicy",
        "env": env,
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,  # Encourage exploration
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "verbose": 1,
        "seed": seed,
        "device": device,
        "tensorboard_log": str(output_dir / "tensorboard"),
    }

    # Policy network architecture
    policy_kwargs = {
        "net_arch": dict(
            pi=[256, 128, 64],  # Actor network
            vf=[256, 128, 64],  # Critic network
        ),
        "activation_fn": torch.nn.ReLU,
    }
    ppo_params["policy_kwargs"] = policy_kwargs

    print("Creating PPO model...")
    model = PPO(**ppo_params)

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=str(output_dir / "checkpoints"),
        name_prefix="ppo_trading",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
    )

    callbacks = CallbackList([checkpoint_callback, eval_callback])

    # Train
    print(f"\nStarting training for {total_timesteps:,} timesteps...")
    print(f"Using device: {device}")

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # Save final model
    final_model_path = output_dir / "final_model"
    model.save(str(final_model_path))
    print(f"\nSaved final model to: {final_model_path}")

    # Cleanup
    env.close()
    eval_env.close()

    return model, output_dir


def evaluate(model_path: str, data_dir: str = None, n_episodes: int = 10):
    """
    Evaluate trained model.

    Args:
        model_path: Path to saved model
        data_dir: Directory with preprocessed data
        n_episodes: Number of evaluation episodes
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data" / "processed"
    else:
        data_dir = Path(data_dir)

    # Load data
    state_array, price_array = load_data(data_dir)

    # Create environment
    env_config = EnvConfig(
        state_dim=state_array.shape[1],
        max_steps=1000,
        random_start=True,
    )
    env = TickTradingEnv(state_array, price_array, env_config)

    # Load model
    model = PPO.load(model_path)
    print(f"Loaded model from: {model_path}")

    # Evaluate
    episode_rewards = []
    episode_trades = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        stats = info["stats"]
        episode_rewards.append(total_reward)
        episode_trades.append(stats["num_trades"])

        print(f"Episode {ep+1}: Reward={total_reward:.4f}, Trades={stats['num_trades']}, "
              f"PnL={stats['total_pnl']:.4f}, WinRate={stats['win_rate']:.2%}")

    print(f"\nEvaluation Summary ({n_episodes} episodes):")
    print(f"  Mean Reward: {np.mean(episode_rewards):.4f} (+/- {np.std(episode_rewards):.4f})")
    print(f"  Mean Trades: {np.mean(episode_trades):.1f}")

    env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train PPO trading agent")
    parser.add_argument("--data-dir", type=str, default=None, help="Data directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--timesteps", type=int, default=500_000, help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval", type=str, default=None, help="Evaluate model at path instead of training")

    args = parser.parse_args()

    if args.eval:
        evaluate(args.eval, args.data_dir)
    else:
        train(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            seed=args.seed,
        )
