# VARnet-lite Market Intelligence Agent v0.1

這是一個金融時間序列觀察 Agent 原型，目標不是直接喊買賣，而是每天自動整理市場資料，偵測趨勢、動能、量能、波動、相對強弱、傅立葉週期與小波異常，產生每日 Markdown 報告。

## 核心定位

- v0.1：每日異常偵測與市場觀察報告
- v0.2：未來 1 日 / 5 日高低區間預測
- v0.3：觸價機率模型
- v0.4：策略回測與 paper trading

## 目前觀察池

台股：2330.TW 台積電、2317.TW 鴻海、2382.TW 廣達、TAIEX 加權指數、TX 台指期

美股：NVDA、TSLA、AMD

加密貨幣：BTC-USD

## 快速開始

```bash
cd finance-varnet-lite
python -m venv .venv
source .venv/bin/activate  # Windows 改用 .venv\\Scripts\\activate
pip install -r requirements.txt
python -m src.main --mode mock
```

執行後會在：

```text
data/reports/YYYY-MM-DD_market_report.md
```

產生一份示範報告。

## 使用真實資料

若要使用 yfinance 作為第一版資料來源：

```bash
python -m src.main --mode yfinance
```

注意：台指期、台股官方資料、三大法人、選擇權、新聞等來源在 v0.1 中先預留接口，後續可新增 fetcher 與 normalizer。

真實資料來源的目前狀態、缺口與台股接入順序請見 `DATA_SOURCE_ROADMAP.md`。

## 設計原則

1. 資料來源可替換，模型只吃標準化資料。
2. 不偷看未來資料，回測需使用 walk-forward validation。
3. v0.1 只做觀察與異常偵測，不產生買賣建議。
4. 所有報告需區分「訊號」、「解讀」與「風險」。
5. 預留多年資料、跨市場資料、新聞事件、期貨選擇權與高低區間預測欄位。

## 專案結構

```text
finance-varnet-lite/
├── AGENTS.md
├── config/
│   ├── watchlist.yaml
│   ├── data_sources.yaml
│   ├── feature_config.yaml
│   └── scoring_rules.yaml
├── src/
│   ├── fetchers/
│   ├── normalizers/
│   ├── features/
│   ├── scoring/
│   ├── report/
│   └── main.py
└── data/
    ├── raw/
    ├── normalized/
    ├── features/
    ├── models/
    └── reports/
```

## v0.2 資料範圍

- 美國股票：NVDA、TSLA、AMD
- 加密貨幣：BTC-USD
- 台灣股票：2330.TW、2317.TW、2382.TW
- 台灣加權指數：TAIEX（yfinance 代號 `^TWII`）
- 總體風險：VIX、美國 10 年期公債殖利率、美元指數、USD/TWD
- 可選資料：TX 台指期、台灣三大法人、融資融券、選擇權與新聞

價格來源會依 `config/data_sources.yaml` 的 primary / fallback 順序切換。
完整分層與欄位說明請見 `DATA_ARCHITECTURE.md`。

報告只提供市場觀察、風險分數、異常訊號與未來區間估計，不提供買賣建議。
