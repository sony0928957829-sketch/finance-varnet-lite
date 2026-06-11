# AGENTS.md

## Project
VARnet-lite Market Intelligence Agent.

## Main Goal
Build a daily financial time-series monitoring system. It should detect abnormal market behavior, trend shifts, volatility expansion, volume anomalies, relative strength, cross-market divergence, and event risks.

## Important Rule
Do not directly produce buy/sell orders. Output observations, risk levels, and watch points only.

## Watchlist
Taiwan:
- 2330.TW TSMC
- 2317.TW Hon Hai
- 2382.TW Quanta
- TAIEX Taiwan Capitalization Weighted Stock Index
- TX Taiwan Index Futures

US:
- NVDA
- TSLA
- AMD

Crypto:
- BTC-USD

## Daily Workflow
1. Fetch OHLCV and relevant market data.
2. Normalize data into the standard price schema.
3. Calculate trend, momentum, volume, volatility, relative strength, Fourier, and wavelet features.
4. Score each symbol.
5. Generate a Markdown daily report under data/reports/.
6. Never use future data for current-day feature calculation.

## Future Workflow
- Add chip data: foreign/investment trust/dealer net buy, margin, short balance.
- Add derivatives: futures basis, OI, option Put/Call ratio.
- Add macro: VIX, DXY, 10Y yield, USD/TWD.
- Add event data: company news, earnings, guidance, macro calendar.
- Add range prediction labels for next 1d, 5d, 10d high/low percentages.

## Coding Style
- Keep data fetchers independent from feature and scoring logic.
- Add a new fetcher for every new data source.
- Use normalizers to convert raw source-specific fields to standard schemas.
- Prefer pandas DataFrame interfaces.
- Include timestamps and source names in outputs.

## Safety and Evaluation
- Always separate signal from interpretation.
- Always report uncertainty.
- Use walk-forward validation for predictive models.
- Include transaction costs if converting signals into strategies.
