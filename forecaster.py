"""Timestral-28M forecasting client, fallback baseline, backtests and chart payload.

The forecast engine returns quantile paths (P10..P90) over a 30-step horizon.
If the Timestral backend is not configured or errors for a metal, that metal
falls back to a transparent random-walk quantile baseline and is labelled so.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from statistics import NormalDist

import streamlit as st

import config
from config import DisplayMode, quantile_label
from data_fetcher import display_unit, get_http_session, to_display

ENGINE_TIMESTRAL = config.TIMESTRAL_MODEL_NAME
ENGINE_BASELINE = "baseline-random-walk"


class ForecastBackendError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Timestral-28M HTTP client
# --------------------------------------------------------------------------- #
class TimestralClient:
    """Thin client for the Timestral-28M zero-shot backend.

    Assumed contract (adapt `_build_request` / `_parse_response` if yours differs):

        POST {TIMESTRAL_API_URL}/v1/forecast
        {"model": "timestral-28m", "context": [float, ...],
         "horizon": 30, "quantiles": [0.1, 0.25, 0.5, 0.75, 0.9], "freq": "B"}

        -> {"quantiles": {"0.1": [..30..], "0.5": [...], ...}}
           or {"forecast": [{"quantile": 0.1, "values": [...]}, ...]}
    """

    def __init__(self, url: str | None, api_key: str | None, model: str):
        self.url = url.rstrip("/") if url else None
        self.api_key = api_key
        self.model = model

    @property
    def available(self) -> bool:
        return bool(self.url)

    def _build_request(self, context: np.ndarray, horizon: int, quantiles: tuple[float, ...]) -> dict:
        return {
            "model": self.model,
            "context": [float(x) for x in context],
            "horizon": int(horizon),
            "quantiles": list(quantiles),
            "freq": config.FORECAST_FREQ,
        }

    @staticmethod
    def _parse_response(payload: dict, horizon: int, quantiles: tuple[float, ...]) -> dict[float, np.ndarray]:
        out: dict[float, np.ndarray] = {}
        if isinstance(payload.get("quantiles"), dict):
            for k, v in payload["quantiles"].items():
                out[round(float(k), 4)] = np.asarray(v, dtype=float)
        elif isinstance(payload.get("forecast"), list):
            for item in payload["forecast"]:
                out[round(float(item["quantile"]), 4)] = np.asarray(item["values"], dtype=float)
        missing = [q for q in quantiles if round(q, 4) not in out]
        if missing:
            raise ForecastBackendError(f"Backend response missing quantiles {missing}")
        result = {q: out[round(q, 4)][:horizon] for q in quantiles}
        if any(len(v) < horizon or not np.all(np.isfinite(v)) for v in result.values()):
            raise ForecastBackendError("Backend returned short or non-finite paths")
        return result

    def predict(self, context: np.ndarray, horizon: int, quantiles: tuple[float, ...]) -> dict[float, np.ndarray]:
        if not self.available:
            raise ForecastBackendError("TIMESTRAL_API_URL not configured")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            resp = get_http_session().post(
                f"{self.url}/v1/forecast",
                json=self._build_request(context, horizon, quantiles),
                headers=headers,
                timeout=config.HTTP_TIMEOUT_SECONDS * 3,
            )
        except Exception as exc:
            raise ForecastBackendError(f"Network error: {type(exc).__name__}") from exc
        if resp.status_code >= 400:
            raise ForecastBackendError(f"HTTP {resp.status_code}")
        try:
            return self._parse_response(resp.json(), horizon, quantiles)
        except (ValueError, KeyError, TypeError) as exc:
            raise ForecastBackendError(f"Unparseable response: {exc}") from exc


# --------------------------------------------------------------------------- #
# Fallback baseline
# --------------------------------------------------------------------------- #
class BaselineQuantileForecaster:
    """Zero-drift log random walk with volatility from recent returns.

    Used only when Timestral is unavailable. Deliberately simple so its
    quantiles are easy to explain and act as a sanity benchmark.
    """

    def __init__(self, vol_window: int = 120):
        self.vol_window = vol_window

    def predict(self, context: np.ndarray, horizon: int, quantiles: tuple[float, ...]) -> dict[float, np.ndarray]:
        context = np.asarray(context, dtype=float)
        rets = np.diff(np.log(context[-(self.vol_window + 1):]))
        sigma = float(np.std(rets, ddof=1)) if len(rets) > 2 else 0.01
        steps = np.arange(1, horizon + 1)
        last = context[-1]
        return {q: last * np.exp(_norm_ppf(q) * sigma * np.sqrt(steps)) for q in quantiles}


def _norm_ppf(q: float) -> float:
    return NormalDist().inv_cdf(q)


# --------------------------------------------------------------------------- #
# Engine (cached once per process)
# --------------------------------------------------------------------------- #
class ForecastEngine:
    def __init__(self) -> None:
        s = config.timestral_settings()
        self.client = TimestralClient(s["url"], s["api_key"], s["model"] or ENGINE_TIMESTRAL)
        self.baseline = BaselineQuantileForecaster()

    @property
    def primary_name(self) -> str:
        return ENGINE_TIMESTRAL if self.client.available else ENGINE_BASELINE

    def predict(self, context: np.ndarray, horizon: int, quantiles: tuple[float, ...]) -> tuple[dict[float, np.ndarray], str, str]:
        """Returns (quantile paths, engine used, note)."""
        context = np.asarray(context, dtype=float)[-config.CONTEXT_LENGTH:]
        note = ""
        if self.client.available:
            try:
                paths = self.client.predict(context, horizon, quantiles)
                return _enforce_monotone(paths, quantiles), ENGINE_TIMESTRAL, "OK"
            except ForecastBackendError as exc:
                note = f"Timestral unavailable ({exc}); baseline used"
        else:
            note = "Timestral not configured; baseline used"
        paths = self.baseline.predict(context, horizon, quantiles)
        return _enforce_monotone(paths, quantiles), ENGINE_BASELINE, note


@st.cache_resource(show_spinner=False)
def get_forecast_engine() -> ForecastEngine:
    return ForecastEngine()


def _enforce_monotone(paths: dict[float, np.ndarray], quantiles: tuple[float, ...]) -> dict[float, np.ndarray]:
    """Sort across quantiles at each step so P10 <= P50 <= P90 always holds."""
    qs = sorted(quantiles)
    stacked = np.sort(np.vstack([paths[q] for q in qs]), axis=0)
    return {q: stacked[i] for i, q in enumerate(qs)}


# --------------------------------------------------------------------------- #
# Cached forecast + backtest
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=config.CACHE_TTL_SECONDS, show_spinner=False)
def forecast_metal(
    metal_key: str,
    context: tuple[float, ...],
    last_date: str,
    horizon: int = config.FORECAST_HORIZON,
    quantiles: tuple[float, ...] = config.QUANTILES,
    refresh_nonce: int = 0,
) -> dict[str, Any]:
    """30-step quantile forecast for one metal (native units)."""
    engine = get_forecast_engine()
    paths, engine_used, note = engine.predict(np.asarray(context), horizon, quantiles)
    dates = pd.bdate_range(pd.Timestamp(last_date) + pd.offsets.BDay(1), periods=horizon)
    start_value = float(context[-1])
    return {
        "metal": metal_key,
        "start_date": last_date,
        "start_value": start_value,
        "dates": [d.date().isoformat() for d in dates],
        "quantiles": {quantile_label(q): paths[q].round(6).tolist() for q in quantiles},
        "engine": engine_used,
        "engine_note": note,
        "horizon": horizon,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


@st.cache_data(ttl=config.CACHE_TTL_SECONDS, show_spinner=False)
def backtest_metal(
    metal_key: str,
    values: tuple[float, ...],
    horizon: int = config.FORECAST_HORIZON,
    quantiles: tuple[float, ...] = config.QUANTILES,
    n_origins: int = config.BACKTEST_ORIGINS,
    refresh_nonce: int = 0,
) -> dict[str, Any]:
    """Rolling-origin backtest: forecast from N past origins, score vs realized."""
    arr = np.asarray(values, dtype=float)
    engine = get_forecast_engine()
    qs = tuple(sorted(quantiles))
    lo_q, hi_q, mid_q = min(qs), max(qs), 0.5 if 0.5 in qs else qs[len(qs) // 2]

    abs_pct_err, dir_hits, dir_total, inside, n_points = [], 0, 0, 0, 0
    below_counts = {q: 0 for q in qs}
    engines_used: set[str] = set()
    windows = []

    for k in range(n_origins, 0, -1):
        origin = len(arr) - k * horizon
        if origin < config.MIN_CONTEXT_POINTS:
            continue
        ctx, actual = arr[:origin], arr[origin:origin + horizon]
        if len(actual) < horizon:
            continue
        paths, eng, _ = engine.predict(ctx, horizon, qs)
        engines_used.add(eng)
        p50, p_lo, p_hi = paths[mid_q], paths[lo_q], paths[hi_q]
        base = ctx[-1]

        abs_pct_err.extend(np.abs(p50 - actual) / np.abs(actual) * 100)
        pred_dir, act_dir = np.sign(p50 - base), np.sign(actual - base)
        mask = (pred_dir != 0) & (act_dir != 0)
        dir_hits += int(np.sum(pred_dir[mask] == act_dir[mask]))
        dir_total += int(np.sum(mask))
        inside += int(np.sum((actual >= p_lo) & (actual <= p_hi)))
        n_points += horizon
        for q in qs:
            below_counts[q] += int(np.sum(actual <= paths[q]))
        windows.append({"origin_index": int(origin), "mape": float(np.mean(np.abs(p50 - actual) / actual * 100))})

    if n_points == 0:
        return {"metal": metal_key, "available": False, "reason": "Not enough history for backtest"}

    return {
        "metal": metal_key,
        "available": True,
        "n_windows": len(windows),
        "n_points": n_points,
        "engines": sorted(engines_used),
        "mape": float(np.mean(abs_pct_err)),
        "directional_hit_rate": (dir_hits / dir_total * 100) if dir_total else None,
        "coverage": inside / n_points * 100,
        "nominal_coverage": (hi_q - lo_q) * 100,
        "band": f"{quantile_label(lo_q).upper()}–{quantile_label(hi_q).upper()}",
        "calibration": [
            {"quantile": q, "label": quantile_label(q).upper(),
             "nominal": q * 100, "observed": below_counts[q] / n_points * 100}
            for q in qs
        ],
        "windows": windows,
    }


# --------------------------------------------------------------------------- #
# Structured chart payload (Streamlit now, Bklit UI props later)
# --------------------------------------------------------------------------- #
def build_chart_payload(
    histories: dict[str, dict],
    forecasts: dict[str, dict],
    metal_keys: list[str],
    mode: str,
    history_points: int,
    horizon_days: int,
    band: tuple[str, str] | None = ("p10", "p90"),
) -> dict[str, Any]:
    """Single source of truth for the chart.

    Plotly renders from this dict today; the same dict can be serialized to JSON
    and passed as props to a React (@bklitui/ui) chart component later.
    """
    series_out = []
    forecast_start = None
    native_units = {config.METALS[k].native_unit for k in metal_keys}
    dual_axis = mode == DisplayMode.NATIVE and len(native_units) > 1
    if mode == DisplayMode.NATIVE:
        left_unit = config.Unit.USD_MT if config.Unit.USD_MT in native_units else config.Unit.USD_OZT
    elif mode == DisplayMode.USD_KG:
        left_unit = config.Unit.USD_KG
    else:
        left_unit = "Index (forecast start = 100)"
    for key in metal_keys:
        spec = config.METALS[key]
        hist = histories[key]["series"].iloc[-history_points:]
        fc = forecasts[key]
        base = fc["start_value"]
        forecast_start = forecast_start or fc["start_date"]

        hist_vals = to_display(hist.values, spec, mode, base)
        fq = {lbl: to_display(np.asarray(v[:horizon_days]), spec, mode, base) for lbl, v in fc["quantiles"].items()}
        axis = "right" if (dual_axis and spec.native_unit == config.Unit.USD_OZT) else "left"

        forecast_rows = []
        for i, d in enumerate(fc["dates"][:horizon_days]):
            row = {"date": d}
            row.update({lbl: round(float(vals[i]), 4) for lbl, vals in fq.items()})
            forecast_rows.append(row)

        series_out.append({
            "id": key,
            "label": spec.name,
            "color": spec.color,
            "unit": display_unit(spec, mode),
            "nativeUnit": spec.native_unit,
            "benchmark": spec.benchmark,
            "axis": axis,
            "status": histories[key]["status"],
            "engine": fc["engine"],
            "startValueNative": round(base, 4),
            "history": [{"date": d.date().isoformat(), "value": round(float(v), 4)}
                        for d, v in zip(hist.index, hist_vals)],
            "forecast": forecast_rows,
        })

    return {
        "schema": "commodity-forecast-chart/v1",
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "displayMode": mode,
        "frequency": config.FORECAST_FREQ,
        "horizon": horizon_days,
        "forecastStart": forecast_start,
        "axes": {"left": left_unit, "right": config.Unit.USD_OZT if dual_axis else None},
        "band": {"lower": band[0], "upper": band[1]} if band else None,
        "median": "p50",
        "series": series_out,
    }
