# VARnet-lite 真實資料來源路線圖

## 原則

- 每個外部來源使用獨立 fetcher，來源欄位由 normalizer 轉為標準價格格式。
- feature、scoring 與 report 不依賴特定供應商。
- 原始資料保留來源、抓取時間與是否還原，方便追蹤品質與切換備援。
- 未來高低區間欄位只作訓練標籤，不作同日訊號或買賣建議。

## 目前狀態

| 市場/資料 | 標的或內容 | 目前來源 | 狀態 |
|---|---|---|---|
| 美股日線 OHLCV | NVDA、TSLA、AMD | yfinance | 已接入 |
| 加密貨幣日線 OHLCV | BTC-USD | yfinance | 已接入 |
| 台股日線 OHLCV | 2330.TW、2317.TW、2382.TW | yfinance | 原型可用，待官方來源備援 |
| 加權指數日線 | TAIEX 對應 `^TWII` | yfinance | 原型可用，待官方來源備援 |
| 台指期 | TX | 無真實來源 | 待接 TAIFEX |

## 尚缺真實資料

1. 台指期 TX：日線與盤中 OHLCV、到期月份、連續合約換月規則、未平倉量與基差。
2. 台股官方行情：TWSE/TPEx 個股與加權指數，作為 yfinance 的驗證與備援來源。
3. 籌碼資料：三大法人、融資融券、借券與證券出借。
4. 衍生品資料：TAIFEX 期貨未平倉、選擇權 Put/Call ratio、履約價分布與波動率。
5. 總體與跨市場資料：VIX、DXY、美國 10 年期殖利率、USD/TWD。
6. 事件資料：財報、法說、公司公告、除權息、經濟日曆與新聞事件。
7. 加密貨幣備援：交易所或 CoinGecko 資料，用來補足 24/7 市場與驗證 yfinance。

## 台股接入順序

### 第一階段：價格原型

- 保留 yfinance 支援 2330.TW、2317.TW、2382.TW 與 `TAIEX -> ^TWII`。
- 每日檢查最新交易日、資料筆數、空值、重複資料與價格跳空。
- yfinance 僅作快速原型，不視為台灣官方行情來源。

### 第二階段：TWSE/TPEx

- 新增 `TwseFetcher`，負責上市個股與加權指數日資料。
- 若要納入上櫃標的，再新增 `TpexFetcher`，不把兩個來源混在同一 fetcher。
- 來源特有欄位先保留在 raw 層，再由 normalizer 映射至標準 OHLCV schema。
- 以交易日、還原方式與成交量單位對照 yfinance，建立資料品質報告。

### 第三階段：TAIFEX 台指期

- 新增 `TaifexFetcher`，先支援 TX 每日行情，再擴充盤中資料。
- 原始資料必須保留契約月份，不能直接把不同月份當成同一時間序列。
- 另建連續合約流程，明確記錄換月規則與調整方式。
- 擴充欄位包含結算價、未平倉量、期現貨基差，之後再接選擇權 Put/Call。

### 第四階段：FinMind 備援與籌碼

- 新增 `FinMindFetcher` 作為台股、法人、融資融券與部分衍生品資料的統一備援。
- 不讓 FinMind 特有欄位進入 feature 層，先經各自 normalizer。
- fetcher factory 後續改為依市場與資料類型路由，而不是只接受單一全域 mode。

## 建議的擴充介面

```text
source config
  -> source-specific fetcher
  -> raw cache
  -> source-specific normalizer
  -> standard price/chip/derivatives/event schema
  -> feature engine
  -> scoring and range-label pipeline
  -> observation report
```

未來預測模型應使用 walk-forward validation，輸出高低區間、觸價機率與不確定性，不輸出直接買賣指令。
