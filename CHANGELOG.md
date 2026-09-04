# 變更紀錄

本檔記錄每次可運行版本的變更、原因與影響。日期為台北時間。

## v0.2.1 — 每日流程穩定性修復（2026-09-04）

### 問題

`Daily Market Report` 排程從 **2026-08-13 之後每天失敗**（連續 20+ 次），
儀表板與報告自該日起停止更新。

失敗訊息（Actions run #180，`Run market report` 步驟）：

```
['NVDA']: ModuleNotFoundError("No module named 'sklearn'")
...
DataHealthError: Price data health check failed: NVDA: Expected symbol has no rows; ...
```

### 根本原因

1. `src/fetchers/yfinance_fetcher.py` 以 `repair=True` 呼叫 `yf.download()`。
   該路徑在 `yfinance/scrapers/history.py` 內 `from sklearn.cluster import DBSCAN`。
2. **yfinance 並未把 scikit-learn 列為相依套件**，而 `requirements.txt` 也沒有裝它。
3. `requirements.txt` 寫 `yfinance>=0.2`（無上限），CI 每次都安裝最新版；
   新版 yfinance 觸發了這條修補路徑，於是**每一檔**下載都丟出 ModuleNotFoundError。
4. 下載全空 → 健康檢查判定 `missing_symbol` → `raise_for_health_errors()` 中止整個流程。
   程式碼未變更卻突然壞掉，正是「未鎖版本的相依套件」典型症狀。

### 修正

| 層級 | 變更 |
| --- | --- |
| 相依套件 | `requirements.txt` 加入 `scikit-learn>=1.3,<2.0`；為 yfinance／pandas／numpy／scipy／PyWavelets 補上版本上限，避免上游改版無聲破壞排程。 |
| 抓取層 | `YFinanceFetcher._download()`：每檔重試 3 次（指數退避），並在 `repair=True` 失敗時自動降級為 `repair=False`；偵測不到 sklearn 時直接跳過修補路徑而非讓整檔消失。 |
| 健康檢查 | 新增 `degraded_mode`：單一標的的「可用性」問題（缺資料／筆數不足／過期／跳空）不再中止全流程；只要涵蓋率 ≥ 60% 且關鍵標的（2330.TW、TAIEX）健在，就以降級模式產出報告。**資料完整性**問題（OHLC 錯亂、空值、負價、重複 K 棒、欄位缺失、全空）仍然硬失敗。 |
| 報告 | 降級運行會在「今日自動摘要」標示涵蓋率與缺漏標的，不會靜默假裝資料完整。 |
| CI | 新增 `Verify runtime dependencies` 步驟，缺套件時 2 秒內明確報錯；報告步驟改為記錄結束碼，失敗時仍上傳 artifact、提交健康報告、並把**實際錯誤訊息**寫進 GitHub Issue（先前只有一行網址）。 |
| CI | 失敗時仍提交健康報告，順帶維持 repo 活動，避免 GitHub 因 60 天無活動自動停用排程。 |

### 測試

新增 `tests/test_pipeline_resilience.py`（10 項）：相依鎖定、repair 降級、
重試、降級模式的四種邊界（部分缺漏可續跑／關鍵標的缺失仍失敗／涵蓋率不足仍失敗／
資料損毀仍失敗）。全套 96 項測試通過。

### 風險與取捨

- 降級模式使「部分資料缺漏」不再中止流程，代價是報告可能少幾檔標的；
  已在報告與 Actions summary 明確標示，且關鍵標的與資料完整性仍為硬性條件。
- 版本上限需要定期人工檢視；建議每季檢查一次上游新版。

## v0.2.0（2026-06-16，回溯記錄）

- 加入成本感知 top-k 回測、每週驗證摘要、可書籤化的 `DASHBOARD.md`。
- 三大法人資料改為 per-metric z-score，消除單位不一致問題。
- 每次執行歸檔至資料湖，並可同步 Google Drive。

## v0.1.0（2026-06，回溯記錄）

- 每日市場觀察報告原型：趨勢、動能、量能、波動、相對強弱、傅立葉週期、小波異常。
