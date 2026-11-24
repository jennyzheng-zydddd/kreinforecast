"""Price data layer: live API pulls, per-metal fallback, unit normalization.

Fallback chain per metal (each metal degrades on its own):
    1. LIVE       configured price API
    2. CACHED     last successful live pull persisted to disk
    3. REFERENCE  deterministic synthetic series (clearly labelled, not market data)
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import config
from config import KG_PER_UNIT, DisplayMode, MetalSpec, Unit

STATUS_LIVE = "live"
STATUS_CACHED = "cached"
STATUS_REFERENCE = "reference"


class ProviderError(RuntimeError):
    """Raised when the price API cannot serve a symbol (plan, rate limit, outage)."""


# --------------------------------------------------------------------------- #
# Shared resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_http_session() -> requests.Session:
    """One pooled HTTP session for the whole server process."""
    session = requests.Session()
    retry = Retry(
        total=2,
        backoff_factor=0.5,
        status_forcelist=(500, 502, 503, 504),  # 429 is NOT retried: degrade instead
        allowed_methods=frozenset({"GET", "POST"}),
    )
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=16)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "commodity-forecast-streamlit/1.0"})
    return session


# --------------------------------------------------------------------------- #
# Live provider (Metals-API style timeseries endpoint)
# --------------------------------------------------------------------------- #
def _parse_rate(day_rates: dict[str, Any], spec: MetalSpec) -> float | None:
    """Return price in `spec.api_unit` for one day of the provider payload."""
    direct = day_rates.get(f"USD{spec.api_symbol}")
    if direct is not None:
        return float(direct)
    raw = day_rates.get(spec.api_symbol)
    if raw in (None, 0):
        return None
    raw = float(raw)
    return 1.0 / raw if spec.api_inverted else raw


def _fetch_live(spec: MetalSpec, start: date, end: date) -> pd.Series:
    settings = config.metals_api_settings()
    if not settings["api_key"]:
        raise ProviderError("No METALS_API_KEY configured")

    session = get_http_session()
    points: dict[pd.Timestamp, float] = {}
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=config.API_MAX_SPAN_DAYS - 1), end)
        try:
            resp = session.get(
                f"{settings['base_url'].rstrip('/')}/timeseries",
                params={
                    "access_key": settings["api_key"],
                    "base": "USD",
                    "symbols": spec.api_symbol,
                    "start_date": chunk_start.isoformat(),
                    "end_date": chunk_end.isoformat(),
                },
                timeout=config.HTTP_TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            raise ProviderError(f"Network error: {type(exc).__name__}") from exc

        if resp.status_code == 429:
            raise ProviderError("Rate limited (HTTP 429)")
        if resp.status_code >= 400:
            raise ProviderError(f"HTTP {resp.status_code}")
        try:
            payload = resp.json()
        except ValueError as exc:
            raise ProviderError("Non-JSON response") from exc
        if payload.get("success") is False:
            err = payload.get("error") or {}
            raise ProviderError(str(err.get("info") or err.get("type") or "Provider error"))

        for day, day_rates in (payload.get("rates") or {}).items():
            if not isinstance(day_rates, dict):
                continue
            price = _parse_rate(day_rates, spec)
            if price is not None and np.isfinite(price) and price > 0:
                points[pd.Timestamp(day)] = price
        chunk_start = chunk_end + timedelta(days=1)

    if len(points) < config.MIN_CONTEXT_POINTS // 2:
        raise ProviderError(f"Symbol {spec.api_symbol} returned too few points ({len(points)})")

    series = pd.Series(points, dtype=float).sort_index()
    return convert_unit(series, spec.api_unit, spec.native_unit)


# --------------------------------------------------------------------------- #
# Disk cache of last good live pull
# --------------------------------------------------------------------------- #
def _disk_path(spec: MetalSpec) -> Path:
    return Path(config.DISK_CACHE_DIR) / f"{spec.key}.csv"


def _write_disk_cache(spec: MetalSpec, series: pd.Series) -> None:
    try:
        path = _disk_path(spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        series.rename("price").to_csv(path, index_label="date")
    except OSError:
        pass  # read-only filesystems (e.g. some cloud hosts) just skip persistence


def _read_disk_cache(spec: MetalSpec) -> tuple[pd.Series, datetime] | None:
    path = _disk_path(spec)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, parse_dates=["date"], index_col="date")
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        return df["price"].astype(float), mtime
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Reference (synthetic) series
# --------------------------------------------------------------------------- #
def _reference_series(spec: MetalSpec, start: date, end: date) -> pd.Series:
    """Deterministic geometric random walk so the UI works offline / without keys.

    Seeded from the metal key only, so it is stable across reruns and sessions.
    """
    idx = pd.bdate_range(start, end)
    seed = int(hashlib.sha256(spec.key.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    shocks = rng.standard_t(df=5, size=len(idx)) * spec.reference_daily_vol * 0.775
    path = spec.reference_anchor * np.exp(np.cumsum(shocks) - shocks.cumsum()[-1])
    return pd.Series(path, index=idx, dtype=float)


# --------------------------------------------------------------------------- #
# Public fetch API
# --------------------------------------------------------------------------- #
def _to_business_days(series: pd.Series) -> pd.Series:
    series = series[~series.index.duplicated(keep="last")].sort_index()
    return series.asfreq(config.FORECAST_FREQ).ffill().dropna()


@st.cache_data(ttl=config.CACHE_TTL_SECONDS, show_spinner=False)
def fetch_history(metal_key: str, lookback_days: int, refresh_nonce: int) -> dict[str, Any]:
    """Historical + latest price series for one metal, in its native unit.

    `refresh_nonce` is part of the cache key only; bumping it forces a new pull.
    Never raises: failures degrade to cached or reference data with a status.
    """
    spec = config.METALS[metal_key]
    end = date.today()
    start = end - timedelta(days=lookback_days)
    now = datetime.now(timezone.utc)

    try:
        series = _to_business_days(_fetch_live(spec, start, end))
        _write_disk_cache(spec, series)
        status, source, as_of, message = STATUS_LIVE, "Price API", now, "Live pull OK"
    except ProviderError as exc:
        cached = _read_disk_cache(spec)
        if cached is not None and len(cached[0]) >= config.MIN_CONTEXT_POINTS:
            series = _to_business_days(cached[0])
            status, source, as_of = STATUS_CACHED, "Disk cache (last live pull)", cached[1]
            message = f"Live unavailable: {exc}"
        else:
            series = _reference_series(spec, start, end)
            status, source, as_of = STATUS_REFERENCE, "Synthetic reference (not market data)", now
            message = f"Live unavailable: {exc}"

    return {
        "metal": metal_key,
        "series": series,
        "status": status,
        "source": source,
        "message": message,
        "as_of": as_of.isoformat(timespec="seconds"),
        "last_obs_date": series.index[-1].date().isoformat(),
        "fetched_at": now.isoformat(timespec="seconds"),
        "unit": spec.native_unit,
    }


def load_all(metal_keys: list[str], lookback_days: int, refresh_nonce: int) -> dict[str, dict]:
    """Fetch every metal independently; one failure never blocks the others."""
    out: dict[str, dict] = {}
    for key in metal_keys:
        try:
            out[key] = fetch_history(key, lookback_days, refresh_nonce)
        except Exception as exc:  # last-resort guard; fetch_history should not raise
            spec = config.METALS[key]
            end = date.today()
            series = _reference_series(spec, end - timedelta(days=lookback_days), end)
            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            out[key] = {
                "metal": key, "series": series, "status": STATUS_REFERENCE,
                "source": "Synthetic reference (not market data)",
                "message": f"Unexpected error: {type(exc).__name__}",
                "as_of": now, "fetched_at": now, "unit": spec.native_unit,
                "last_obs_date": series.index[-1].date().isoformat(),
            }
    return out


def clear_price_cache() -> None:
    fetch_history.clear()


# --------------------------------------------------------------------------- #
# Unit normalization
# --------------------------------------------------------------------------- #
def convert_unit(values, from_unit: str, to_unit: str):
    """Convert a price-per-unit between USD/MT, USD/troy oz and USD/kg."""
    if from_unit == to_unit:
        return values
    factor = KG_PER_UNIT[to_unit] / KG_PER_UNIT[from_unit]
    return values * factor


def display_unit(spec: MetalSpec, mode: str) -> str:
    if mode == DisplayMode.INDEXED:
        return "Index (forecast start = 100)"
    if mode == DisplayMode.USD_KG:
        return Unit.USD_KG
    return spec.native_unit


def to_display(values, spec: MetalSpec, mode: str, base_value: float):
    """Transform native-unit prices to the chosen display mode.

    `base_value` is the native-unit price at forecast start (last observation).
    """
    arr = np.asarray(values, dtype=float)
    if mode == DisplayMode.INDEXED:
        return arr / base_value * 100.0
    if mode == DisplayMode.USD_KG:
        return convert_unit(arr, spec.native_unit, Unit.USD_KG)
    return arr


def provenance_snapshot(results: dict[str, dict]) -> dict[str, Any]:
    """Compact, JSON-serializable provenance block for UI and exports."""
    return {
        key: {k: r[k] for k in ("status", "source", "message", "as_of", "last_obs_date", "unit")}
        for key, r in results.items()
    }


def provenance_json(results: dict[str, dict]) -> str:
    return json.dumps(provenance_snapshot(results), indent=2)
