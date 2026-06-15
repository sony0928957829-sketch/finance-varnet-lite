# VARnet-lite Data Architecture

VARnet-lite separates source collection, normalization, features, labels, models,
and reports. A source-specific field must not leak directly into feature or
report code.

## Storage layers

| Path | Purpose |
|---|---|
| `data/raw/` | Optional source-native cache |
| `data/normalized/` | Standard OHLCV price observations |
| `data/alternative/` | News, chip, margin, futures, and option observations |
| `data/features/` | Live-safe features and range forecast fields |
| `data/labels/` | Future labels for training and backtests only |
| `data/models/` | Serialized future model artifacts |
| `data/reports/` | Markdown reports and health JSON |

Generated Parquet files are retained as GitHub Actions artifacts. Markdown and
health reports are committed by the daily workflow.

## Standard price schema

`datetime, symbol, market, timeframe, open, high, low, close, volume, source,
adjusted, created_at`

Canonical symbols remain stable across providers. Provider aliases are declared
in `config/data_sources.yaml`, for example `TAIEX -> ^TWII` for yfinance.

## Alternative schemas

- Chip data uses long-form `dataset, metric, value, unit` rows.
- Derivatives use contract, expiry, option type, strike, OHLC, settlement,
  volume, open interest, and generic value fields.
- News uses event type, title, summary, URL, publisher, and provenance fields.

Each enabled supplemental route writes its own Parquet file, such as
`chip_taiwan_institutional.parquet` or `derivatives_taiwan_options.parquet`.
Category-level `news.parquet`, `chip.parquet`, and `derivatives.parquet` files
combine successful routes for downstream analysis.

## Provider routing

Every enabled dataset has one primary provider and an ordered fallback list.
The router records each attempt and missing symbol in the supplemental health
report.

The same routing rule applies to alternative data. FinMind supplies
contract-level TXO data and derived Put/Call ratios; TAIFEX is the official
ratio fallback. TWSE provides latest-day chip data when FinMind fails.

| Dataset | Primary | Fallback |
|---|---|---|
| US stocks and BTC | yfinance | Polygon / Alpha Vantage / Binance / CoinGecko |
| Taiwan stocks | yfinance | FinMind / TWSE |
| TAIEX | yfinance | TWSE / FinMind |
| TX futures | TAIFEX | FinMind |
| Taiwan chip data | FinMind | TWSE |
| Taiwan options | FinMind | TAIFEX |
| VIX | yfinance | FRED |
| US 10Y yield | yfinance | FRED |
| US dollar index | yfinance | Stooq |
| USD/TWD | yfinance | FinMind / Taiwan central bank |
| News | Yahoo Finance News | Reuters / CNA / MoneyDJ |

FinMind accepts an optional `FINMIND_TOKEN`. Without a token, the public API
quota applies. External-source errors are recorded without suppressing the core
price report.

## Publication-time alignment

Taiwan institutional, margin, futures-position, and option observations are
published after the cash-market close. A source row dated `t` is therefore
made available to live features from the next business observation date, not
the same date. Taiwan option sentiment is scoped to Taiwan instruments and is
not attached to US stocks or BTC.

## Leakage boundary

`next_1d_*`, `next_5d_*`, and `next_10d_*` are labels and only belong in
`data/labels/`. Live features contain `pred_next_*` fields. The baseline model
shifts each realized label by its horizon before calculating trailing
quantiles, so a forecast cannot use a label that is not yet observable.

## Output policy

Outputs are market observations, risk scores, anomaly signals, data-source
health, and forecast ranges. They are not buy or sell recommendations.
