# Tick RL Trading

東証ティックデータを用いた強化学習デイトレードシステム

## 概要

J-Quants DataCubeのティックデータから100ティックバーを生成し、
PPO（Proximal Policy Optimization）エージェントで自動売買戦略を学習するシステム。

## プロジェクト構造

```
tick_data/
├── stock_tick_data/         # 生ティックデータディレクトリ
│   └── stock_tick_202510.csv    # ティックデータ (8GB)
├── tick_data_spec.md        # データ仕様書
├── pyproject.toml           # 依存関係
├── README.md
│
├── src/
│   ├── data/               # データ処理
│   │   ├── loader.py       # CSVローダー
│   │   ├── bar_aggregator.py   # バー集約
│   │   └── preprocessor.py # 前処理・特徴量計算
│   │
│   ├── features/           # 特徴量計算
│   │   ├── price_features.py   # 価格系
│   │   ├── volume_features.py  # 出来高系
│   │   ├── microstructure.py   # マイクロストラクチャー
│   │   └── technical.py        # テクニカル指標
│   │
│   ├── env/                # 強化学習環境
│   │   ├── trading_env.py  # Gymnasium環境
│   │   ├── actions.py      # 行動空間
│   │   └── reward.py       # 報酬関数
│   │
│   └── agents/             # エージェント
│
├── notebooks/              # Jupyter notebooks
├── scripts/                # 実行スクリプト
├── data/processed/         # 前処理済みデータ
└── models/                 # 学習済みモデル
```

## セットアップ

### 1. 仮想環境作成

```bash
cd /Users/asefujiko/tools/tick_data
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
```

### 2. 依存関係インストール

```bash
pip install -e .
# または
pip install polars numpy torch stable-baselines3[extra] gymnasium matplotlib plotly pyarrow jupyterlab ipywidgets tqdm
```

### 3. CUDA確認 (4070Ti Super)

```bash
# macOS/Linux
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Windows (CMD)
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

## 使い方

### Step 1: データ前処理

```bash
# macOS/Linux
python scripts/preprocess.py --stock 72030

# カスタム設定 (macOS/Linux)
python scripts/preprocess.py \
    --csv /path/to/tick_data.csv \
    --stock 72030 \
    --bar-size 100 \
    --warmup 100 \
    --output ./data/processed

# Windows (CMD)
python scripts\preprocess.py --stock 72030

# カスタム設定 (Windows CMD)
python scripts\preprocess.py ^
    --csv C:\path\to\tick_data.csv ^
    --stock 72030 ^
    --bar-size 100 ^
    --warmup 100 ^
    --output .\data\processed
```

### Step 2: モデル学習

```bash
# macOS/Linux
python scripts/train.py --timesteps 500000

# カスタム設定 (macOS/Linux)
python scripts/train.py \
    --timesteps 1000000 \
    --n-envs 4 \
    --seed 42 \
    --output-dir ./models/experiment1

# Windows (CMD)
python scripts\train.py --timesteps 500000

# カスタム設定 (Windows CMD)
python scripts\train.py ^
    --timesteps 1000000 ^
    --n-envs 4 ^
    --seed 42 ^
    --output-dir .\models\experiment1
```

### Step 3: 評価

```bash
# macOS/Linux
python scripts/train.py --eval ./models/ppo_xxx/final_model

# Windows (CMD)
python scripts\train.py --eval .\models\ppo_xxx\final_model
```

## 特徴量一覧

### 価格系 (Price Features)
- `return_1/5/20/100`: N期間リターン
- `price_position_20/100`: レンジ内位置 [0, 1]
- `realized_vol_20/100`: 実現ボラティリティ
- `vol_ratio`: 短期/長期ボラティリティ比

### 出来高系 (Volume Features)
- `volume_ratio_20/100`: 出来高/移動平均比
- `volume_zscore`: 出来高Zスコア
- `vwap_deviation_20/100`: VWAP乖離率
- `dollar_volume_normalized`: 正規化金額出来高

### マイクロストラクチャー
- `trade_sign`: 取引サイン (+1/-1)
- `ofi_normalized_20/100`: オーダーフロー不均衡
- `cum_sign_normalized_20`: 累積取引サイン

### テクニカル指標
- `rsi_20_normalized`: RSI (正規化)
- `bb_position_20`: ボリンジャーバンド位置
- `bb_width_20`: バンド幅
- `macd_histogram_normalized`: MACDヒストグラム

### 時間特徴量
- `time_sin/cos`: 時刻の周期エンコーディング
- `is_opening`: 寄り付きフラグ
- `is_closing`: 大引けフラグ

## 強化学習設定

### MDP定義
- **State**: 28次元の特徴量ベクトル + 4次元のポジション情報
- **Action**: 離散 {HOLD, BUY, SELL}
- **Reward**: PnL - 取引コスト

### PPOハイパーパラメータ
- Learning rate: 3e-4
- Batch size: 64
- n_steps: 2048
- Entropy coefficient: 0.01

## データ仕様

### 入力データ
- ファイル: `stock_tick_data/stock_tick_202510.csv` (約8GB)
- 期間: 2025年10月
- カラム: date, issue_code, time, price, volume, session等

### 100ティックバー
- 100ティックごとに1バーに集約
- OHLCV + 各種メタデータ

## トラブルシューティング

### メモリ不足
- `bar_size`を大きくする (例: 200)
- 対象銘柄を絞る

### GPU認識しない
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Polarsエラー
```bash
pip install polars --upgrade
```

## ライセンス

Private use only.
