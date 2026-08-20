"""Price loading with an on-disk parquet cache.

Contract: the cache is authoritative. The network is best-effort and is only
touched when the cache cannot cover the requested window. If the network fails,
we serve what the cache has and say so, rather than raising.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

CACHE = Path(__file__).parent / "cache"
MIN_OBS = 120  # ~6 months of trading days; below this, betas and covariances are noise


def _path(ticker: str) -> Path:
    return CACHE / f"{ticker.replace('/', '_')}.parquet"


def _read_cache(ticker: str) -> pd.Series | None:
    p = _path(ticker)
    if not p.exists():
        return None
    try:
        s = pd.read_parquet(p)["close"]
        s.index = pd.to_datetime(s.index)
        return s.sort_index()
    except Exception:
        return None


def _write_cache(ticker: str, s: pd.Series) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    s.to_frame("close").to_parquet(_path(ticker))


def _download(tickers: list[str], start, end) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(
        tickers, start=start, end=end,
        auto_adjust=True, progress=False, group_by="column", threads=True,
    )
    if raw is None or raw.empty:
        return pd.DataFrame()
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    close.columns = [str(c) for c in close.columns]
    return close


def get_prices(
    tickers: list[str],
    start: str | dt.date,
    end: str | dt.date,
    allow_network: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (adjusted close prices, per-ticker quality report).

    Tickers with fewer than MIN_OBS observations in the window are dropped from
    the price frame but stay in the report with `kept=False`, so the UI can show
    what was thrown away and why.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    tickers = list(dict.fromkeys(t.strip().upper() for t in tickers if t.strip()))

    cached = {t: _read_cache(t) for t in tickers}
    # ponytail: if the cache misses at either edge we refetch the whole union window
    # in one call rather than splicing head and tail gaps. One network call, no
    # splice bugs. Revisit only if the universe gets big enough for it to hurt.
    stale = [
        t for t, s in cached.items()
        if s is None or s.index.min() > start or s.index.max() < end - pd.Timedelta(days=5)
    ]

    source = {t: "cache" for t in tickers}
    net_error = None
    if stale and allow_network:
        lo = min([start] + [s.index.min() for s in cached.values() if s is not None])
        hi = max([end] + [s.index.max() for s in cached.values() if s is not None])
        try:
            fresh = _download(stale, lo, hi)
            for t in stale:
                if t in fresh.columns:
                    s = fresh[t].dropna()
                    if len(s):
                        _write_cache(t, s)
                        cached[t] = s
                        source[t] = "network"
        except Exception as exc:  # offline, rate-limited, ticker delisted
            net_error = f"{type(exc).__name__}: {exc}"

    frame = pd.DataFrame({t: s for t, s in cached.items() if s is not None})
    if frame.empty:
        return frame, pd.DataFrame(columns=["ticker", "rows", "nans", "first", "last", "source", "kept"])

    frame = frame.loc[(frame.index >= start) & (frame.index <= end)].sort_index()

    rows = []
    for t in tickers:
        col = frame[t] if t in frame.columns else pd.Series(dtype=float)
        obs = int(col.notna().sum())
        rows.append({
            "ticker": t,
            "rows": obs,
            "nans": int(col.isna().sum()),
            "first": col.dropna().index.min() if obs else pd.NaT,
            "last": col.dropna().index.max() if obs else pd.NaT,
            "source": source.get(t, "missing"),
            "kept": obs >= MIN_OBS,
        })
    report = pd.DataFrame(rows)
    if net_error:
        report.attrs["network_error"] = net_error

    keep = [r["ticker"] for r in rows if r["kept"]]
    # forward-fill holidays that differ across exchanges, then drop leading gaps
    return frame[keep].ffill().dropna(how="any"), report


def get_market_caps(tickers: list[str], allow_network: bool = True) -> dict[str, float] | None:
    """Market caps for the Black-Litterman prior. None if unavailable — callers
    must fall back to a flat prior and say so in the UI.

    Cached to disk because each lookup is a separate `.info` request: fifteen
    tickers is fifteen round trips and it blocks the page every rerun otherwise.
    Caps move slowly; a stale one is a far smaller error than the flat prior
    that a failed lookup forces.
    """
    import json

    CACHE.mkdir(parents=True, exist_ok=True)
    store = CACHE / "market_caps.json"
    known = {}
    if store.exists():
        try:
            known = json.loads(store.read_text())
        except Exception:
            known = {}

    missing = [t for t in tickers if t not in known]
    if missing and allow_network:
        try:
            import yfinance as yf

            for t in missing:
                mc = yf.Ticker(t).info.get("marketCap")
                if mc:
                    known[t] = float(mc)
            store.write_text(json.dumps(known, indent=1))
        except Exception:
            pass

    caps = {t: known[t] for t in tickers if t in known}
    return caps if len(caps) == len(tickers) else None
