# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

東証ティックデータを用いた強化学習デイトレードシステム（Tick RL Trading）

J-Quants DataCubeのティックデータから100ティックバーを生成し、PPO（Proximal Policy Optimization）エージェントで自動売買戦略を学習する。

## Project Structure

```
tick_data/
├── stock_tick_data/           # 生ティックデータ (gitignore)
│   └── stock_tick_202510.csv  # 8GB CSVファイル
├── data/processed/            # 前処理済みデータ
├── models/                    # 学習済みモデル
│
├── src/
│   ├── data/                  # データ処理モジュール
│   │   ├── loader.py          # Polars CSVローダー
│   │   ├── bar_aggregator.py  # ティックバー集約
│   │   └── preprocessor.py    # 特徴量計算オーケストレーション
│   │
│   ├── features/              # 特徴量計算
│   │   ├── price_features.py      # 価格系特徴量
│   │   ├── volume_features.py     # 出来高系特徴量
│   │   ├── microstructure.py      # マイクロストラクチャー
│   │   └── technical.py           # テクニカル指標
│   │
│   └── env/                   # 強化学習環境
│       ├── trading_env.py     # Gymnasium環境
│       ├── actions.py         # 行動空間定義
│       └── reward.py          # 報酬関数
│
├── scripts/
│   ├── preprocess.py          # 前処理パイプライン
│   └── train.py               # PPO学習スクリプト
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
└── docs/
    └── requirements_specification.md  # 要件定義書
```

## Environment Setup

### Windows

```cmd
cd C:\Users\fujikko\devml\tick_data
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

### Mac/Linux

```bash
cd /Users/asefujiko/tools/tick_data
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### GPU確認 (4070Ti Super)

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## Common Commands

### データ前処理

```bash
# トヨタ(7203)のデータを100ティックバーに変換
python scripts/preprocess.py --stock 72030

# カスタム設定
python scripts/preprocess.py \
    --csv ./stock_tick_data/stock_tick_202510.csv \
    --stock 72030 \
    --bar-size 100 \
    --warmup 100
```

### モデル学習

```bash
# 基本学習
python scripts/train.py --timesteps 500000

# GPU並列環境
python scripts/train.py --timesteps 1000000 --n-envs 4 --seed 42
```

### 評価

```bash
python scripts/train.py --eval ./models/ppo_xxx/final_model
```

## Technical Architecture

### MDP Formulation

| Component | Definition |
|-----------|------------|
| **State** | 32次元 = 28市場特徴量 + 4ポジション情報 |
| **Action** | 離散3値: HOLD(0), BUY(1), SELL(2) |
| **Reward** | PnL - 取引コスト(0.1%) |

### Feature Engineering (28 Features)

1. **価格系**: リターン(1,5,20,100期間), 実現ボラティリティ, 価格位置
2. **出来高系**: 出来高比率, VWAP乖離, 金額出来高
3. **マイクロストラクチャー**: Lee-Ready取引サイン, OFI, 累積サイン
4. **テクニカル**: RSI, ボリンジャーバンド, MACD
5. **時間特徴量**: sin/cos周期エンコーディング, 寄付/大引けフラグ

### PPO Hyperparameters

```python
learning_rate = 3e-4
n_steps = 2048
batch_size = 64
n_epochs = 10
gamma = 0.99
gae_lambda = 0.95
clip_range = 0.2
ent_coef = 0.01
```

## Data Specifications

### Input: J-Quants DataCube Tick Data

- 15フィールド: date, issue_code, time(μs精度), price, volume, session等
- 約8500万行/月
- Polars lazy evaluationで効率的に処理

### Output: 100-Tick Bars

- 100ティックごとに1バーに集約
- OHLCV + メタデータ（tick_count, duration_seconds, trade_date等）

## Key Design Decisions

1. **100ティックバー選択**: 時間バーより情報効率が高く、ボリュームバーより実装が単純
2. **PPO選択**: TRPO/A2Cより安定、DQNより連続的な行動調整に適応
3. **離散行動空間**: 連続行動よりサンプル効率が良い
4. **Lee-Ready法**: 気配値なしでも取引サインを推定可能
5. **Z-score正規化**: [-3, 3]でクリップし外れ値の影響を抑制

## Dependencies

- **Data**: polars, pyarrow, numpy
- **ML**: torch (CUDA 12.1), stable-baselines3, gymnasium
- **Viz**: matplotlib, plotly
- **Dev**: jupyterlab, ipywidgets, tqdm

## Notes

- 8GB CSVは直接読まず、Polars lazy + stock_code filteringで処理
- warmup期間（デフォルト100バー）は特徴量計算で使用、学習データからは除外
- GPU 4070Ti Superで学習高速化可能
