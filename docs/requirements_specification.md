# 要件定義書：東証ティックデータ強化学習デイトレードシステム

**文書バージョン**: 1.0
**作成日**: 2025-12-01
**ステータス**: 実装完了（Phase 0-4）

---

## 1. プロジェクト概要

### 1.1 背景と目的

O'Reilly書籍「Hands-On Machine Learning for Algorithmic Trading」の「7章 時系列分析：機械学習によるデイトレード」で紹介される知見を活用し、J-Quants DataCubeの東証ティックデータから**強化学習（Reinforcement Learning）**による自動売買戦略を学習するシステムを構築する。

### 1.2 プロジェクトスコープ

| 項目 | 範囲 |
|------|------|
| **データソース** | J-Quants DataCube 歩み値（ティック）データ |
| **対象市場** | 東京証券取引所（立会内取引のみ） |
| **対象期間** | 2025年10月（1ヶ月分、約8GB） |
| **初期対象銘柄** | トヨタ自動車（7203）- 流動性の高い銘柄 |
| **手法** | 強化学習（PPO: Proximal Policy Optimization） |
| **目標** | ランダムエージェントを上回る取引戦略の学習 |

### 1.3 設計方針の決定プロセス

開発にあたり、以下の設計判断を行った：

| 検討事項 | 選択肢 | 採用 | 採用理由 |
|---------|--------|------|---------|
| **初期スコープ** | 全銘柄 vs 単一銘柄 | 単一銘柄 | 小さく実験的に開始し、成功パターンを確立 |
| **時間スケール** | 生ティック / 時間バー / ティックバー / ボリュームバー | 100ティックバー | 計算効率と情報量のバランスが最良 |
| **行動空間** | 離散（3択）/ 離散（11択）/ 連続 | 離散3択 | シンプルから始めて段階的に拡張 |
| **アルゴリズム** | DQN / PPO / SAC / TD3 | PPO | 安定性とサンプル効率のバランスが最良 |
| **報酬関数** | シンプルPnL / リスク調整 / 多目的 | シンプルPnL | まず基本動作を確認、後に複雑化 |

---

## 2. データ仕様

### 2.1 入力データ項目

J-Quants DataCube「株式・CB 歩み値 20210802以降のフォーマット」に準拠。

| No. | 項目名（英語） | 項目名（日本語） | データ型 | 説明 |
|-----|--------------|----------------|---------|------|
| 1 | Date | 年月日 | YYYYMMDD | 取引日 |
| 2 | Issue code | 銘柄コード | 5桁 | 証券コード協議会定義 |
| 3 | ISIN code | ISINコード | 12桁 | 国際証券識別番号 |
| 4 | Exchange code | 執行市場 | 2桁 | 01：東証 |
| 5 | Issue classification | 銘柄区分 | 5桁 | 銘柄区分コード |
| 6 | Industry code | 業種コード | 4桁 | 業種分類 |
| 7 | Supervision flag | 整理・監理銘柄区分 | 1桁 | 1:整理、2:監理、空白:その他 |
| 8 | Time | 成立時刻 | hhmmsstttttt | **マイクロ秒精度** |
| 9 | Session distinction | 場区分 | 2桁 | 01:前場、02:後場 |
| 10 | Price | 歩み値 | 小数4桁 | 約定価格（円） |
| 11 | Trading volume | 取引高 | 整数 | 約定株数 |
| 12 | Transaction ID | トランザクションID | 文字列 | 同時刻約定の順序識別 |

### 2.2 データ留意事項

- **株式分割未調整**: 調整係数の適用が必要な場合あり
- **立会内取引のみ**: ToSTNeT取引等は含まれない
- **JASDAQ統合**: 2013/7/16以降は東証データに統合

### 2.3 対象銘柄の選定基準

| 基準 | 理由 |
|------|------|
| 高流動性 | 十分なティック数でモデル学習が可能 |
| 大型株 | 価格操作リスクが低い |
| 業種代表 | 一般化可能性の検証 |

**選定結果**: トヨタ自動車（7203）
- ティック数: 約361,592（1ヶ月）
- 業種コード: 3700（輸送用機器）
- 価格帯: 約2,800円

---

## 3. システム設計

### 3.1 アーキテクチャ概要

```
┌─────────────────────────────────────────────────────────────────┐
│                    Tick RL Trading System                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Raw CSV    │───▶│  100-Tick    │───▶│   Feature    │      │
│  │   (8GB)      │    │    Bars      │    │   Vector     │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌──────────────────────────────────────────────────────┐      │
│  │              Gymnasium Environment                    │      │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐              │      │
│  │  │  State  │  │ Action  │  │ Reward  │              │      │
│  │  │ (32dim) │  │ (3択)   │  │ (PnL)   │              │      │
│  │  └─────────┘  └─────────┘  └─────────┘              │      │
│  └──────────────────────────────────────────────────────┘      │
│                            │                                    │
│                            ▼                                    │
│  ┌──────────────────────────────────────────────────────┐      │
│  │              PPO Agent (Stable-Baselines3)            │      │
│  │  ┌─────────────┐         ┌─────────────┐            │      │
│  │  │   Actor     │         │   Critic    │            │      │
│  │  │ (256-128-64)│         │ (256-128-64)│            │      │
│  │  └─────────────┘         └─────────────┘            │      │
│  └──────────────────────────────────────────────────────┘      │
│                            │                                    │
│                            ▼                                    │
│  ┌──────────────────────────────────────────────────────┐      │
│  │         Evaluation & Backtesting                      │      │
│  └──────────────────────────────────────────────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 モジュール構成

```
tick_data/
├── src/
│   ├── data/                    # データ処理層
│   │   ├── loader.py           # CSVローダー（Polars）
│   │   ├── bar_aggregator.py   # バー集約
│   │   └── preprocessor.py     # 特徴量計算・正規化
│   │
│   ├── features/                # 特徴量計算層
│   │   ├── price_features.py   # 価格系（リターン、ボラティリティ）
│   │   ├── volume_features.py  # 出来高系（VWAP、OFI）
│   │   ├── microstructure.py   # マイクロストラクチャー
│   │   └── technical.py        # テクニカル指標
│   │
│   ├── env/                     # 強化学習環境層
│   │   ├── trading_env.py      # Gymnasium環境
│   │   ├── actions.py          # 行動空間定義
│   │   └── reward.py           # 報酬関数
│   │
│   └── agents/                  # エージェント層
│
├── scripts/                     # 実行スクリプト
│   ├── preprocess.py           # 前処理パイプライン
│   └── train.py                # 学習実行
│
├── notebooks/                   # 実験ノートブック
├── data/processed/             # 前処理済みデータ
└── models/                     # 学習済みモデル
```

---

## 4. 強化学習フレームワーク設計

### 4.1 MDP（マルコフ決定過程）定義

強化学習の問題をMDPとして定式化：

```
MDP = (S, A, P, R, γ)

S: 状態空間 - 市場特徴量 + ポジション情報
A: 行動空間 - {HOLD, BUY, SELL}
P: 遷移確率 - 市場の確率的動態（環境から与えられる）
R: 報酬関数 - PnL - 取引コスト
γ: 割引率 - 0.99（将来報酬の重み）
```

### 4.2 状態空間（State Space）設計

#### 4.2.1 市場特徴量（28次元）

| カテゴリ | 特徴量 | 次元 | 説明 |
|---------|--------|------|------|
| **価格動態** | return_1, return_5, return_20, return_100 | 4 | 複数期間のリターン |
| | price_position_20, price_position_100 | 2 | レンジ内位置 [0,1] |
| **ボラティリティ** | realized_vol_20, realized_vol_100 | 2 | 実現ボラティリティ |
| | vol_ratio | 1 | 短期/長期ボラ比 |
| **出来高** | volume_ratio_20, volume_ratio_100 | 2 | 出来高/MA比 |
| | volume_zscore | 1 | 異常出来高検出 |
| | vwap_deviation_20, vwap_deviation_100 | 2 | VWAP乖離率 |
| | dollar_volume_normalized | 1 | 正規化金額出来高 |
| **オーダーフロー** | ofi_normalized_20, ofi_normalized_100 | 2 | Order Flow Imbalance |
| | cum_sign_normalized_20 | 1 | 累積取引サイン |
| | trade_sign | 1 | 直近取引サイン |
| **テクニカル** | rsi_20_normalized | 1 | RSI（正規化） |
| | bb_position_20 | 1 | BBバンド内位置 |
| | bb_width_20 | 1 | BBバンド幅 |
| | macd_histogram_normalized | 1 | MACDヒストグラム |
| **時間** | time_sin, time_cos | 2 | 時刻の周期エンコーディング |
| | is_opening, is_closing | 2 | 寄り/引けフラグ |

#### 4.2.2 ポジション情報（4次元）

| 特徴量 | 説明 |
|--------|------|
| position | 現在ポジション（-1/0/1） |
| unrealized_pnl | 含み損益（正規化） |
| holding_time | 保有バー数（正規化） |
| entry_price_normalized | エントリー価格（正規化） |

**合計**: 32次元

#### 4.2.3 正規化戦略

```python
# Z-score正規化（推奨）
normalized = (value - rolling_mean) / (rolling_std + ε)

# クリッピング
normalized = clip(normalized, -5.0, 5.0)
```

**理論的根拠**:
- Z-scoreは平均0、分散1に正規化し、ニューラルネットワークの学習を安定化
- クリッピングは外れ値による勾配爆発を防止

### 4.3 行動空間（Action Space）設計

#### 4.3.1 採用: 離散3択

```python
actions = {
    0: 'HOLD',   # 現状維持
    1: 'BUY',    # ロングエントリー or ショートクローズ
    2: 'SELL',   # ショートエントリー or ロングクローズ
}
```

#### 4.3.2 行動解釈ロジック

| 現在ポジション | BUY実行 | SELL実行 |
|---------------|---------|----------|
| ショート (-1) | → フラット (0) | 変化なし |
| フラット (0) | → ロング (+1) | → ショート (-1) |
| ロング (+1) | 変化なし | → フラット (0) |

#### 4.3.3 将来の拡張オプション

```python
# 拡張5択（ロットサイズ制御）
actions_extended = {
    0: 'HOLD',
    1: 'BUY_SMALL',   # 10%ポジション
    2: 'BUY_LARGE',   # 50%ポジション
    3: 'SELL_SMALL',
    4: 'SELL_LARGE',
}

# 連続行動空間（PPO/SAC向け）
action = Box(low=-1.0, high=1.0, shape=(1,))
# -1.0 = フルショート, 0 = フラット, +1.0 = フルロング
```

### 4.4 報酬関数（Reward Function）設計

#### 4.4.1 採用: シンプルPnL報酬

```python
def compute_reward(position, price_change, trade_direction, price, config):
    # PnL（ポジション × 価格変化率）
    pnl = position * price_change

    # 取引コスト（0.1%）
    transaction_cost = abs(trade_direction) * price * config.transaction_cost

    return pnl - transaction_cost
```

#### 4.4.2 報酬設計の理論的背景

**シンプルPnLを採用した理由**:
1. **解釈可能性**: 報酬が実際の損益に直結
2. **デバッグ容易性**: 異常動作の原因特定が容易
3. **段階的複雑化**: まず基本動作を確認後、拡張可能

**避けるべきパターン**:
```python
# ❌ 取引ペナルティが大きすぎる → 何もしないエージェントに
reward = pnl - 0.1 * num_trades

# ❌ 負の報酬を無視 → 損切りできないエージェントに
reward = max(pnl, 0)

# ❌ 短期報酬のみ → 長期的なドローダウン無視
reward = immediate_pnl  # 累積影響なし
```

#### 4.4.3 将来の拡張: リスク調整報酬

```python
def risk_adjusted_reward(pnl, volatility, position, config):
    # シャープレシオ的な報酬
    base_reward = pnl
    vol_penalty = -config.vol_coef * volatility * abs(position)
    drawdown_penalty = config.dd_coef * min(pnl, 0) ** 2

    return base_reward + vol_penalty - drawdown_penalty
```

### 4.5 エピソード設計

| 項目 | 設定 | 理由 |
|------|------|------|
| 最大ステップ数 | 1,000バー | 1日の取引を概ねカバー |
| 開始位置 | ランダム | 過学習防止 |
| 終了条件 | 最大ステップ or データ終端 | 自然な区切り |
| エピソード終了時 | ポジション強制クローズ | リスク管理 |

---

## 5. 特徴量エンジニアリング設計

### 5.1 特徴量の理論的背景

#### 5.1.1 第1層：価格特徴量

**リターン系列**
```python
r_t = (P_t - P_{t-1}) / P_{t-1}       # 単純リターン
log_r_t = log(P_t / P_{t-1})          # 対数リターン
```

**理論的根拠**:
- 対数リターンは加法性を持つ（複数期間の合計が容易）
- 価格水準に依存しない正規化された変動を捉える
- 統計的性質が単純リターンより良好（より正規分布に近い）

**ローリング統計量**
```python
rolling_mean(N)  = mean(r_{t-N:t})
rolling_std(N)   = std(r_{t-N:t})    # = 実現ボラティリティ
```

**窓サイズの意味**:
| 窓サイズ | 意味 | 用途 |
|---------|------|------|
| 20バー | 約2,000ティック | 短期トレンド検出 |
| 100バー | 約10,000ティック | 中期トレンド検出 |

**価格レンジ位置**
```python
position_in_range = (P_t - min_N) / (max_N - min_N)
# 0 = レンジ下限, 1 = レンジ上限
```

**意義**: 「買われすぎ/売られすぎ」の定量化（逆張りシグナル）

#### 5.1.2 第2層：出来高特徴量

**VWAP（出来高加重平均価格）**
```python
VWAP_N = Σ(P_i × V_i) / Σ(V_i)
VWAP_deviation = (P_t - VWAP_N) / VWAP_N
```

**トレーディング意義**:
- 機関投資家のベンチマーク価格
- VWAPを上回る → 買い圧力優勢
- VWAPを下回る → 売り圧力優勢

**出来高Zスコア**
```python
volume_zscore = (V_t - mean(V)) / std(V)
```

**意義**: 異常出来高の検出（ニュースイベント、大口注文の兆候）

#### 5.1.3 第3層：マイクロストラクチャー特徴量

**取引サイン推定（Lee-Ready法）**
```python
if P_t > P_{t-1}:
    trade_sign = +1   # 買い（アップティック）
elif P_t < P_{t-1}:
    trade_sign = -1   # 売り（ダウンティック）
else:
    trade_sign = trade_sign_{t-1}  # 直前を継承
```

**理論的背景**:
- Lee & Ready (1991) の古典的手法
- 板情報がない場合の売買推定として広く使用
- 約85%の精度で実際の売買方向を推定

**Order Flow Imbalance (OFI)**
```python
OFI_N = Σ(trade_sign_i × V_i)  # 符号付き出来高の累積
normalized_OFI = OFI_N / Σ(V_i)
```

**意義**:
- 正値 = 買い圧力優勢（価格上昇圧力）
- 負値 = 売り圧力優勢（価格下落圧力）
- 短期的な価格変動の先行指標

#### 5.1.4 第4層：時間特徴量

**周期性エンコーディング**
```python
time_sin = sin(2π × minutes_from_open / total_trading_minutes)
time_cos = cos(2π × minutes_from_open / total_trading_minutes)
```

**理論的根拠**:
- 時刻を連続値としてエンコード
- sin/cosペアで周期性を表現（0時と24時が近い値になる）
- ニューラルネットワークが日内パターンを学習しやすい

**重要時間帯フラグ**
```python
is_opening = (9:00 <= time < 9:05)   # 寄り付き直後
is_closing = (14:50 <= time < 15:00) # 大引け前
```

**意義**:
- 寄り付き・大引けは特殊な価格形成プロセス
- ボラティリティと流動性が通常時と異なる

#### 5.1.5 第5層：ボラティリティ特徴量

**実現ボラティリティ**
```python
RV_N = sqrt(Σ(r_i²))  # 二乗リターンの平方根
```

**理論的背景**:
- Andersen & Bollerslev (1998) による高頻度データからのボラティリティ推定
- 日中データを使用することで、日次データより精度の高い推定が可能

**ボラティリティ比**
```python
vol_ratio = realized_vol_20 / realized_vol_100
```

**意義**:
- vol_ratio > 1: ボラティリティ上昇中（市場活性化）
- vol_ratio < 1: ボラティリティ低下中（市場沈静化）

#### 5.1.6 第6層：テクニカル指標

**RSI（相対力指数）**
```python
RS = avg_gain / avg_loss
RSI = 100 - 100 / (1 + RS)
```

**正規化**:
```python
RSI_normalized = (RSI - 50) / 50  # [-1, 1]に変換
```

**ボリンジャーバンド位置**
```python
BB_position = (P_t - BB_lower) / (BB_upper - BB_lower)
# 0 = 下限バンド, 1 = 上限バンド
```

**MACD**
```python
MACD_line = EMA_12 - EMA_26
Signal = EMA_9(MACD_line)
Histogram = MACD_line - Signal
```

### 5.2 特徴量選択の設計判断

| 採用した特徴量 | 理由 |
|---------------|------|
| リターン系列 | 価格変動の基本情報 |
| 実現ボラティリティ | リスク測定に必須 |
| VWAP乖離 | 機関投資家行動の代理変数 |
| OFI | 短期価格予測に有効 |
| RSI/BB | 過熱/過冷の検出 |
| 時間特徴量 | 日内パターン捕捉 |

| 見送った特徴量 | 理由 |
|---------------|------|
| ハースト指数 | 計算コスト高、効果未検証 |
| エントロピー | 解釈困難、過学習リスク |
| クロスアセット | 単一銘柄フェーズでは不要 |
| VPIN | 実装複雑、効果未検証 |

---

## 6. 技術スタック

### 6.1 開発環境

| 項目 | 仕様 |
|------|------|
| OS | macOS Darwin 24.6.0 |
| Python | 3.11+ |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER |
| CUDA | 12.x |

### 6.2 主要ライブラリ

| ライブラリ | バージョン | 用途 | 選定理由 |
|-----------|----------|------|---------|
| polars | 1.0+ | データ処理 | pandasより10x高速、メモリ効率良好 |
| numpy | 1.26+ | 数値計算 | 標準的な数値計算ライブラリ |
| torch | 2.0+ | 深層学習 | CUDA対応、SB3との互換性 |
| stable-baselines3 | 2.0+ | 強化学習 | 実績豊富、使いやすいAPI |
| gymnasium | 0.29+ | RL環境 | OpenAI Gym後継、標準インターフェース |
| pyarrow | 14.0+ | Parquet | 高速I/O、列指向ストレージ |

### 6.3 pyproject.toml

```toml
[project]
name = "tick-rl-trading"
version = "0.1.0"
requires-python = ">=3.11"

dependencies = [
    "polars>=1.0.0",
    "numpy>=1.26.0",
    "torch>=2.0.0",
    "stable-baselines3[extra]>=2.0.0",
    "gymnasium>=0.29.0",
    "matplotlib>=3.8.0",
    "plotly>=5.18.0",
    "pyarrow>=14.0.0",
    "jupyterlab>=4.0.0",
    "ipywidgets>=8.1.0",
    "tqdm>=4.66.0",
]
```

---

## 7. 実装仕様

### 7.1 データ処理フロー

```
[Raw CSV 8GB]
      │
      ▼ loader.py
[Filtered by Stock Code]  # トヨタ: 361,592 ticks
      │
      ▼ bar_aggregator.py
[100-Tick Bars]           # 3,615 bars
      │
      ▼ preprocessor.py
[Feature Matrix]          # (3,515, 28) after warmup
      │
      ▼
[State Array .npy]        # 学習用データ
```

### 7.2 環境インターフェース

```python
class TickTradingEnv(gym.Env):
    """
    Gymnasium互換の取引環境

    Observation: np.ndarray, shape=(32,), dtype=float32
        - 市場特徴量 (28次元)
        - ポジション情報 (4次元)

    Action: int in {0, 1, 2}
        - 0: HOLD
        - 1: BUY
        - 2: SELL

    Reward: float
        - PnL - transaction_cost

    Terminated: bool
        - 常にFalse（早期終了なし）

    Truncated: bool
        - True if step >= max_steps
    """
```

### 7.3 PPOハイパーパラメータ

```python
ppo_params = {
    "learning_rate": 3e-4,      # 学習率
    "n_steps": 2048,            # 更新あたりのステップ数
    "batch_size": 64,           # ミニバッチサイズ
    "n_epochs": 10,             # エポック数
    "gamma": 0.99,              # 割引率
    "gae_lambda": 0.95,         # GAEパラメータ
    "clip_range": 0.2,          # PPOクリッピング範囲
    "ent_coef": 0.01,           # エントロピー係数（探索促進）
    "vf_coef": 0.5,             # 価値関数係数
    "max_grad_norm": 0.5,       # 勾配クリッピング
}

policy_kwargs = {
    "net_arch": {
        "pi": [256, 128, 64],   # Actor（方策）ネットワーク
        "vf": [256, 128, 64],   # Critic（価値）ネットワーク
    },
    "activation_fn": torch.nn.ReLU,
}
```

---

## 8. 非機能要件

### 8.1 性能要件

| 項目 | 要件 |
|------|------|
| データ読み込み | 8GB CSV を 5分以内に処理 |
| 特徴量計算 | 36万ティックを 1分以内に処理 |
| 学習速度 | GPU使用時 10,000 steps/分 以上 |
| 推論速度 | 1ステップ < 1ms |

### 8.2 メモリ要件

| フェーズ | 推定メモリ使用量 |
|---------|----------------|
| CSV読み込み（Polars lazy） | < 4GB |
| バー集約後 | < 100MB |
| 特徴量計算後 | < 50MB |
| 学習時（バッチ） | < 2GB GPU VRAM |

### 8.3 拡張性要件

| 項目 | 対応 |
|------|------|
| 複数銘柄対応 | MultiAssetTradingEnv クラスを用意 |
| 行動空間拡張 | actions.py で設定変更可能 |
| 報酬関数変更 | RewardConfig で設定変更可能 |
| 特徴量追加 | features/ 配下にモジュール追加 |

---

## 9. 実装ステータス

### 9.1 完了項目

- [x] Phase 0: プロジェクト構造作成
- [x] Phase 0: pyproject.toml作成
- [x] Phase 1: CSVの構造確認
- [x] Phase 1: 単一銘柄データ抽出・100ティックバー集約
- [x] Phase 2: 特徴量エンジニアリング（28次元）
- [x] Phase 3: Gymnasium環境構築
- [x] Phase 4: PPO学習スクリプト

### 9.2 今後の拡張予定

- [ ] 報酬関数の改善（リスク調整版）
- [ ] 複数銘柄での汎化テスト
- [ ] ハイパーパラメータ最適化（Optuna）
- [ ] バックテストフレームワーク構築
- [ ] 本番運用向けリアルタイム推論システム

---

## 10. 付録

### 10.1 アルゴリズム選定の詳細比較

| アルゴリズム | 特徴 | メリット | デメリット | デイトレード適性 |
|------------|------|---------|-----------|----------------|
| **PPO** | Policy Gradient + Clipping | 安定、汎用性高 | サンプル効率中程度 | ★★★★★ |
| **SAC** | 最大エントロピー強化学習 | 探索効率良、連続行動に最適 | 実装やや複雑 | ★★★★☆ |
| **DQN** | Q-learning + Deep NN | 実装容易、離散行動に最適 | 過大評価問題 | ★★★☆☆ |
| **A2C** | Actor-Critic | シンプル、並列化容易 | 分散大 | ★★★☆☆ |
| **TD3** | Twin Delayed DDPG | 過大評価抑制、連続行動 | 決定論的方策 | ★★★★☆ |

**PPO採用理由**:
1. 安定した学習曲線（金融データの非定常性に強い）
2. 離散・連続両方の行動空間に対応
3. Stable-Baselines3での実装が成熟
4. ハイパーパラメータ感度が低い

### 10.2 バー集約方式の比較

| バー種類 | 集約基準 | メリット | デメリット |
|---------|---------|---------|-----------|
| **時間バー** | 固定時間（1分等） | 直感的、従来手法と親和 | 情報量が不均一 |
| **ティックバー** | 固定ティック数 | 情報量が均一 | 時間が不均一 |
| **ボリュームバー** | 固定出来高 | 流動性正規化 | 閑散期にバー生成少 |
| **ドルバー** | 固定金額出来高 | 価格影響正規化 | 計算やや複雑 |

**100ティックバー採用理由**:
1. 情報量が均一化（各バーが同じ「情報単位」）
2. 時間バーより統計的性質が良好
3. 実装がシンプル
4. 計算効率と情報量のバランスが良好

### 10.3 参考文献

1. O'Reilly「Hands-On Machine Learning for Algorithmic Trading」
2. Lee, C. M., & Ready, M. J. (1991). Inferring trade direction from intraday data.
3. Andersen, T. G., & Bollerslev, T. (1998). Answering the skeptics: Yes, standard volatility models do provide accurate forecasts.
4. Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms.
5. J-Quants DataCube Documentation: https://jpx.gitbook.io/jquants-dc/

---

**文書終了**
