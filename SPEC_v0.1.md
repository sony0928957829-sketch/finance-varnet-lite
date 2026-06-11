# VARnet-lite v0.1 規格書

## 1. 專案目標

建立一個每日自主執行的金融市場觀察 Agent，針對指定標的進行時間序列分析，偵測：

- 趨勢狀態
- 動能變化
- 量能異常
- 波動擴大
- 相對強弱
- 傅立葉週期
- 小波異常
- 跨市場背離

v0.1 不產生買賣建議，只輸出觀察、分數與風險提醒。

## 2. 觀察標的

| 分類 | 標的 |
|---|---|
| 台股權值 | 2330.TW 台積電 |
| 台股 AI | 2317.TW 鴻海、2382.TW 廣達 |
| 台股大盤 | TAIEX 加權指數 |
| 台股期貨 | TX 台指期 |
| 美股 AI | NVDA、AMD |
| 美股高波動科技 | TSLA |
| 加密貨幣 | BTC-USD |

## 3. 標準化價格資料格式

```csv
datetime,symbol,market,timeframe,open,high,low,close,volume,source,adjusted,created_at
```

### 欄位說明

| 欄位 | 說明 |
|---|---|
| datetime | 資料時間 |
| symbol | 標的代號 |
| market | TW / US / CRYPTO |
| timeframe | 1d / 1h / 5m 等 |
| open/high/low/close | 開高低收 |
| volume | 成交量 |
| source | 資料來源 |
| adjusted | 是否為還原價格 |
| created_at | 資料建立時間 |

## 4. 資料來源設計

採用可擴充 fetcher 架構：

```text
資料來源 → fetcher → normalizer → feature engine → scoring engine → report generator
```

每新增一個資料來源，只需新增：

1. fetcher
2. normalizer
3. config/data_sources.yaml 設定

核心 feature/scoring 不需要重寫。

## 5. v0.1 特徵

| 特徵 | 用途 |
|---|---|
| return_1d / 5d / 20d / 60d | 短中期報酬 |
| ma_5 / 20 / 60 / 120 / 240 | 趨勢狀態 |
| volume_ratio | 量能是否異常 |
| volatility_20 / 60 | 波動狀態 |
| atr_pct | 真實波動區間 |
| fourier_main_cycle | 主要週期 |
| fourier_cycle_strength | 週期強度 |
| wavelet_anomaly_score | 小波異常分數 |
| relative_strength_20d | 相對強弱 |

## 6. 分數設計

| 分數 | 說明 |
|---|---|
| trend_score | 趨勢健康程度 |
| momentum_score | 動能強弱 |
| volume_score | 量能狀態 |
| volatility_risk_score | 波動風險 |
| relative_strength_score | 相對基準強弱 |
| anomaly_risk_score | 小波與波動異常 |
| condition_score | 綜合狀態，不是買賣訊號 |
| risk_score | 綜合風險分數 |

## 7. 報告輸出

每日輸出 Markdown：

```text
data/reports/YYYY-MM-DD_market_report.md
```

內容包含：

1. 市場總覽
2. 標的分數表
3. 今日異常訊號
4. 相對強弱排序
5. 隔日觀察重點
6. v0.2 高低區間預測預留

## 8. v0.2 高低區間預測預留欄位

```csv
next_1d_high_pct,next_1d_low_pct,next_5d_high_pct,next_5d_low_pct,next_10d_high_pct,next_10d_low_pct
```

這些欄位只能作為訓練標籤，不可作為同日特徵，避免資料洩漏。

## 9. 回測原則

未來若進入預測模型，必須使用 walk-forward validation：

```text
2018-2021 訓練 → 2022 測試
2018-2022 訓練 → 2023 測試
2018-2023 訓練 → 2024 測試
2018-2024 訓練 → 2025 測試
```

不得使用隨機切分，避免未來資料洩漏。

## 10. 後續擴充路線

### v0.2

- 高低區間預測
- 1日、5日、10日未來高低點百分比
- Quantile regression / LightGBM / XGBoost

### v0.3

- 觸價機率模型
- 未來 N 日碰到 +3%、-3%、前高、月線的機率

### v0.4

- 策略回測
- 交易成本、滑價、證交稅、手續費
- Paper trading

### v0.5

- GitHub Actions / cron 定時排程
- Telegram / LINE / Email 自動推送
