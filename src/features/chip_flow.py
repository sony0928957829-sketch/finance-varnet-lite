"""籌碼面 / 選擇權情緒特徵。

把已經被 collect_supplemental_data() 抓回來、存在 data/alternative/ 的籌碼與
選擇權資料，接成可以餵進 scoring 的特徵欄位。

設計重點（為什麼這樣寫）：
1. 不寫死任何資料內的字串值。三大法人在 `metric` 裡實際叫什麼、選擇權的
   `option_type` 是 "call"/"put" 還是 "C"/"P"，這支程式都用「關鍵字比對 +
   自動偵測」去對，對不上就回傳 NaN，不會默默產生假資料。
2. 所有輸出都是 0~100 分，跟現有的 trend_score / momentum_score 同尺度，
   方便直接併進 scores.py 的加權。
3. 找不到檔案 / 欄位 / 可用資料時，整條安靜降級成 NaN，不讓主流程崩潰。

產出的欄位：
    institutional_flow_score   外資/法人買賣超的相對強度（每檔台股，0~100，50=中性）
    put_call_ratio             當日 Put/Call 量比（市場層級，原始值，非分數）
    options_sentiment_score    選擇權情緒（市場層級，0~100，>50 偏多 / <50 偏空）
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TAIWAN_SYMBOLS = {"TAIEX", "TX"}

# 用來在 chip 資料的 metric/dataset 欄位裡，認出「法人淨買賣」這類列。
# 不分大小寫、中英都比。
#
# 注意：這裡刻意「不放」單獨的 net / buy / sell，因為它們會誤命中
# net_income / buyback / sell_side_rating 之類無關的 metric。改成
# 「法人主體」與「明確的淨買賣詞」兩類，命中後在 _institutional_flow_score
# 內再要求兩類至少各有一個（或本身就是明確片語），降低誤抓。
_INSTITUTIONAL_ENTITIES = (
    "foreign", "investment_trust", "trust", "dealer", "institutional",
    "外資", "投信", "自營", "法人", "三大法人",
)
# 明確到本身就足以判定是淨買賣的片語（單獨命中即可採用）。
_INSTITUTIONAL_PHRASES = (
    "net_buy", "net_sell", "net_buy_sell", "buy_sell_net", "net_position",
    "買賣超", "淨買賣", "淨買超", "淨賣超", "買超", "賣超",
)
# 需要搭配「法人主體」才採用的弱動作詞（避免單獨的 buy/sell 誤抓）。
_INSTITUTIONAL_ACTIONS = ("net", "buy", "sell", "買", "賣")


def _normalize_datetime(series: pd.Series) -> pd.Series:
    """把任意 datetime 欄位轉成 tz-naive。

    為什麼不直接用 .dt.tz_localize(None)：對「本來就 tz-naive」的資料，
    tz_localize(None) 會丟 TypeError: Already tz-naive，讓整條流程崩掉，
    違反本模組「安靜降級」的設計。這裡先判斷有沒有 tz，再決定怎麼脫。
    """
    dt = pd.to_datetime(series, errors="coerce")
    tz = getattr(dt.dt, "tz", None)
    if tz is not None:
        dt = dt.dt.tz_convert(None)
    return dt


def _read_alternative(category_prefix: str, alt_dir: Path) -> pd.DataFrame:
    """把 data/alternative/ 底下某類（chip_* / derivatives_*）的 parquet 全部讀進來疊起來。"""
    if not alt_dir.exists():
        return pd.DataFrame()
    frames = []
    for path in sorted(alt_dir.glob(f"{category_prefix}_*.parquet")):
        try:
            frames.append(pd.read_parquet(path))
        except Exception:
            # 單一檔壞掉不該拖垮整條，跳過即可。
            logger.warning("skip unreadable parquet: %s", path, exc_info=True)
            continue
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if "datetime" in out.columns:
        out["datetime"] = _normalize_datetime(out["datetime"])
        out = out.dropna(subset=["datetime"])
    return out


def _rolling_z(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    """滾動 z-score，用來把「今天的買賣超」放到自己過去的分布上看是不是異常。"""
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std(ddof=0)
    return (series - mean) / std.replace(0, np.nan)


def _institutional_mask(hay: pd.Series) -> pd.Series:
    """從文字欄位（已 lower 後）判斷哪幾列是法人淨買賣。

    採用條件（任一成立即採用）：
      A. 命中明確片語（買賣超 / net_buy ...）。
      B. 同時命中「法人主體」與「弱動作詞」（外資 + 買 ...）。
    只命中弱動作詞（單獨 net/buy/sell）不採用，避免誤抓。
    """
    def hit(s: str) -> bool:
        if any(p in s for p in _INSTITUTIONAL_PHRASES):
            return True
        has_entity = any(e in s for e in _INSTITUTIONAL_ENTITIES)
        has_action = any(a in s for a in _INSTITUTIONAL_ACTIONS)
        return has_entity and has_action

    return hay.apply(hit)


def _institutional_flow_score(
    chip: pd.DataFrame,
    *,
    window: int = 60,
    min_periods: int = 20,
) -> pd.DataFrame:
    """每檔台股每天一個 institutional_flow_score（0~100，50=中性）。"""
    cols = [
        "symbol",
        "datetime",
        "institutional_flow_score",
        "available_from",
    ]
    if chip.empty:
        return pd.DataFrame(columns=cols)

    # 自動挑出「像是法人淨買賣」的列：metric 或 dataset 命中規則。
    text_cols = [c for c in ("metric", "dataset") if c in chip.columns]
    if not text_cols or "value" not in chip.columns or "symbol" not in chip.columns:
        return pd.DataFrame(columns=cols)

    hay = chip[text_cols].astype(str).agg(" ".join, axis=1).str.lower()
    mask = _institutional_mask(hay)
    flow = chip.loc[mask].copy()
    if flow.empty:
        return pd.DataFrame(columns=cols)

    # 跨主體加總前的單位/符號健全性檢查。
    # 本步驟「假設」同一 symbol+datetime 下各列已是可相加的同單位、同符號慣例
    # （買超為正、賣超為負）。若上游混用股數與金額，這裡的加總會失真——
    # 無法在資料層面 100% 驗證，至少把假設標明，並偵測明顯異常（量級跨度過大）。
    flow["value"] = pd.to_numeric(flow["value"], errors="coerce")
    flow = flow.dropna(subset=["value"])
    if flow.empty:
        return pd.DataFrame(columns=cols)

    nz = flow["value"].abs()
    nz = nz[nz > 0]
    if not nz.empty:
        span = nz.max() / nz.min()
        if span > 1e6:
            # 量級橫跨百萬倍，極可能是股數與金額混在一起，加總無意義。
            logger.warning(
                "institutional flow values span %.2g orders of magnitude; "
                "likely mixed units (shares vs value). Skipping flow score.",
                span,
            )
            return pd.DataFrame(columns=cols)

    # 同一天同一檔可能有多列（外資/投信/自營），加總成「總淨買賣」。
    daily = (
        flow.groupby(["symbol", "datetime"], as_index=False)["value"]
        .sum()
        .sort_values(["symbol", "datetime"])
    )
    if daily.empty:
        return pd.DataFrame(columns=cols)

    daily["z"] = (
        daily.groupby("symbol")["value"]
        .transform(lambda s: _rolling_z(s, window, min_periods))
    )
    # z 用 logistic 壓到 0~100；z=0 -> 50。係數 0.9 讓 ±2σ 大約落在 ~85 / ~15。
    daily["institutional_flow_score"] = (
        100.0 / (1.0 + np.exp(-0.9 * daily["z"]))
    ).clip(0, 100)
    # Taiwan institutional data is published after the cash close. A date-t
    # observation therefore becomes a feature no earlier than the next
    # business observation date.
    daily["available_from"] = daily["datetime"] + pd.offsets.BDay(1)
    return daily[cols]


def _options_sentiment(derivatives: pd.DataFrame) -> pd.DataFrame:
    """市場層級的 put_call_ratio 與 options_sentiment_score（每個 datetime 一列）。"""
    empty = pd.DataFrame(
        columns=["datetime", "put_call_ratio", "options_sentiment_score"]
    )
    if derivatives.empty or "option_type" not in derivatives.columns:
        return empty

    df = derivatives.copy()
    opt = df["option_type"].astype(str).str.lower()
    is_put = opt.str.startswith("p")
    is_call = opt.str.startswith("c")
    if not is_put.any() or not is_call.any():
        return empty

    # 量能優先用 volume，沒有就退而用 open_interest。
    weight_col = (
        "volume"
        if "volume" in df.columns and df["volume"].notna().any()
        else ("open_interest" if "open_interest" in df.columns else None)
    )
    if weight_col is None:
        return empty

    df[weight_col] = pd.to_numeric(df[weight_col], errors="coerce").fillna(0)
    puts = df.loc[is_put].groupby("datetime")[weight_col].sum()
    calls = df.loc[is_call].groupby("datetime")[weight_col].sum()
    ratio = (
        (puts / calls.replace(0, np.nan))
        .rename("put_call_ratio")
        .to_frame()
        .reset_index()
        .sort_values("datetime")  # 明確排序，確保滾動分位是時間序
    )
    if ratio.empty:
        return empty

    # P/C 高 = 避險/偏空 -> 情緒分數低。用滾動分位把比值轉成 0~100。
    r = ratio["put_call_ratio"]
    rank = r.rolling(120, min_periods=20).apply(
        lambda x: (x.iloc[-1] >= x).mean() * 100.0, raw=False
    )
    ratio["options_sentiment_score"] = (100.0 - rank).clip(0, 100)
    ratio["available_from"] = ratio["datetime"] + pd.offsets.BDay(1)
    return ratio[
        [
            "datetime",
            "available_from",
            "put_call_ratio",
            "options_sentiment_score",
        ]
    ]


def _is_taiwan_instrument(frame: pd.DataFrame) -> pd.Series:
    symbol = frame.get("symbol", pd.Series("", index=frame.index)).astype(str)
    market = frame.get("market", pd.Series("", index=frame.index)).astype(str)
    return (
        symbol.str.endswith(".TW")
        | symbol.isin(TAIWAN_SYMBOLS)
        | market.str.upper().str.startswith("TW")
    )


def add_chip_flow_features(
    frame: pd.DataFrame,
    *,
    alt_dir: Path | str = Path("data/alternative"),
) -> pd.DataFrame:
    """主入口：在 price-feature frame 上補出三個籌碼/選擇權欄位。

    任何一步失敗或無資料，對應欄位都會是 NaN，scores.py 端會把 NaN 當「無此訊號」跳過。
    """
    output = frame.copy()
    for col in ("institutional_flow_score", "put_call_ratio", "options_sentiment_score"):
        output[col] = np.nan
    if output.empty:
        return output

    # 把 frame 的 datetime 也正規化成 tz-naive，避免與 alt 資料 dtype/tz 不一致
    # 導致 merge 全部對不上、整欄靜默變 NaN（這種「假降級」會掩蓋接線錯誤）。
    if "datetime" in output.columns:
        output["datetime"] = _normalize_datetime(output["datetime"])

    alt_dir = Path(alt_dir)
    chip = _read_alternative("chip", alt_dir)
    derivatives = _read_alternative("derivatives", alt_dir)

    flow = _institutional_flow_score(chip)
    if not flow.empty:
        if "symbol" not in output.columns or "datetime" not in output.columns:
            logger.warning(
                "frame missing symbol/datetime; cannot merge institutional flow."
            )
        elif output["datetime"].dtype != flow["datetime"].dtype:
            logger.warning(
                "datetime dtype mismatch (frame=%s, flow=%s); flow merge would "
                "yield all-NaN. Skipping.",
                output["datetime"].dtype, flow["datetime"].dtype,
            )
        else:
            flow_context = flow[
                ["symbol", "available_from", "institutional_flow_score"]
            ].rename(columns={"available_from": "datetime"})
            frames = []
            for symbol, symbol_frame in output.groupby("symbol", sort=False):
                source = flow_context.loc[
                    flow_context["symbol"].eq(symbol),
                    ["datetime", "institutional_flow_score"],
                ]
                left = symbol_frame.drop(
                    columns=["institutional_flow_score"],
                    errors="ignore",
                ).sort_values("datetime")
                if source.empty:
                    left["institutional_flow_score"] = np.nan
                    frames.append(left)
                    continue
                frames.append(
                    pd.merge_asof(
                        left,
                        source.sort_values("datetime"),
                        on="datetime",
                        direction="backward",
                        tolerance=pd.Timedelta(days=7),
                    )
                )
            output = pd.concat(frames, ignore_index=True).sort_values(
                ["symbol", "datetime"]
            )

    sentiment = _options_sentiment(derivatives)
    if not sentiment.empty:
        if "datetime" not in output.columns:
            logger.warning("frame missing datetime; cannot merge options sentiment.")
        elif output["datetime"].dtype != sentiment["datetime"].dtype:
            logger.warning(
                "datetime dtype mismatch (frame=%s, sentiment=%s); sentiment merge "
                "would yield all-NaN. Skipping.",
                output["datetime"].dtype, sentiment["datetime"].dtype,
            )
        else:
            taiwan_mask = _is_taiwan_instrument(output)
            taiwan = output.loc[taiwan_mask].drop(
                columns=["put_call_ratio", "options_sentiment_score"],
                errors="ignore",
            )
            other = output.loc[~taiwan_mask].copy()
            source = sentiment[
                [
                    "available_from",
                    "put_call_ratio",
                    "options_sentiment_score",
                ]
            ].rename(columns={"available_from": "datetime"})
            if not taiwan.empty:
                taiwan = pd.merge_asof(
                    taiwan.sort_values("datetime"),
                    source.sort_values("datetime"),
                    on="datetime",
                    direction="backward",
                    tolerance=pd.Timedelta(days=7),
                )
            output = pd.concat([taiwan, other], ignore_index=True).sort_values(
                ["symbol", "datetime"]
            )

    return output.reset_index(drop=True)
