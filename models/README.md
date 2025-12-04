# Models Directory

PPO強化学習エージェントの学習済みモデルを格納するディレクトリ。

## ディレクトリ構造

```
models/
└── ppo_YYYYMMDD_HHMMSS/     # 学習実行ごとのディレクトリ
    ├── final_model.zip      # 最終モデル（学習完了時）
    ├── best_model/
    │   └── best_model.zip   # 評価スコア最高時のモデル
    ├── checkpoints/         # 定期保存（10,000ステップごと）
    │   ├── ppo_trading_10000_steps.zip
    │   ├── ppo_trading_20000_steps.zip
    │   └── ...
    ├── eval_logs/           # 評価結果
    │   └── evaluations.npz  # 評価スコア履歴
    └── tensorboard/         # TensorBoardログ
```

## モデルの使い方

### 評価の実行

```bash
# best_modelを使用（推奨）
python scripts/train.py --eval models/ppo_YYYYMMDD_HHMMSS/best_model/best_model

# final_modelを使用
python scripts/train.py --eval models/ppo_YYYYMMDD_HHMMSS/final_model
```

### 学習曲線の可視化

```bash
tensorboard --logdir models/ppo_YYYYMMDD_HHMMSS/tensorboard
```

ブラウザで http://localhost:6006 を開く。

### Pythonからモデルをロード

```python
from stable_baselines3 import PPO

# モデルのロード
model = PPO.load("models/ppo_YYYYMMDD_HHMMSS/best_model/best_model")

# 推論
action, _ = model.predict(observation, deterministic=True)
```

## ファイルの説明

| ファイル | 説明 | 用途 |
|----------|------|------|
| `final_model.zip` | 学習完了時点のモデル | 学習が正常終了した場合の最終状態 |
| `best_model.zip` | 評価スコアが最高だった時点のモデル | **本番運用に推奨** |
| `checkpoints/` | 学習途中の定期保存 | 学習再開、ロールバック |
| `eval_logs/` | 評価メトリクスの記録 | 学習進捗の分析 |
| `tensorboard/` | 詳細な学習ログ | 報酬・損失の可視化 |

## best_model vs final_model

- **best_model**: 評価エピソードで最高の平均報酬を達成した時点のモデル。過学習を避けた最適なモデル。
- **final_model**: 指定したステップ数の学習が完了した時点のモデル。必ずしも最高性能とは限らない。

通常は **best_model** を使用することを推奨。

## 学習の再開

チェックポイントから学習を再開する場合（未実装）:

```python
model = PPO.load("models/ppo_xxx/checkpoints/ppo_trading_200000_steps")
model.learn(total_timesteps=300000)  # 追加学習
```

## 注意事項

- モデルファイル（.zip）は `.gitignore` で除外されている
- 大規模な学習結果は外部ストレージにバックアップを推奨
- 異なる銘柄で学習したモデルは互換性がない可能性あり
