"""Commodity Forecasting: Timestral-28M 30-day probabilistic outlook, one metal at a time.

User journey:
    0. Landing      "Choose Metal" opens a pop-up (@st.dialog)
    1. Dialog       pick one metal + view settings (units, risk band, windows), confirm
    2. Foundation   KPIs, benchmark/unit context, provenance for that metal
    3. Forecast     30-day quantile chart built from build_chart_payload()
    4. Risk         backtest metrics, calibration, quantile table, exports

Selection and settings live in st.session_state (and the ?metal= query param),
so they survive reruns and the dialog can be reopened at any time to switch.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from config import METALS, DisplayMode
from data_fetcher import (
    STATUS_CACHED,
    STATUS_LIVE,
    STATUS_REFERENCE,
    clear_price_cache,
    convert_unit,
    fetch_history,
    provenance_snapshot,
)
from forecaster import (
    ENGINE_TIMESTRAL,
    backtest_metal,
    build_chart_payload,
    forecast_metal,
    get_forecast_engine,
)

st.set_page_config(page_title="Metals 30-Day Forecast", page_icon="📈", layout="wide")

STATUS_STYLE = {
    STATUS_LIVE: ("LIVE", "#1E8449"),
    STATUS_CACHED: ("CACHED", "#B9770E"),
    STATUS_REFERENCE: ("REFERENCE", "#7B7D7D"),
}
BAND_OPTIONS: dict[str, tuple[str, str] | None] = {
    "P10–P90 (80% band)": ("p10", "p90"),
    "P25–P75 (50% band)": ("p25", "p75"),
    "No band": None,
}
DEFAULT_VIEW: dict[str, Any] = {
    "mode": DisplayMode.NATIVE,          # single metal: native units read most naturally
    "band": "P10–P90 (80% band)",
    "horizon_days": config.FORECAST_HORIZON,
    "history_points": 120,
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def fmt_price(value: float, unit: str) -> str:
    return f"{value:,.0f} {unit}" if value >= 1000 else f"{value:,.2f} {unit}"


def pct(a: float, b: float) -> float:
    return (b / a - 1.0) * 100.0


def status_badge(status: str) -> str:
    label, color = STATUS_STYLE[status]
    return (f"<span style='background:{color};color:white;border-radius:4px;"
            f"padding:1px 6px;font-size:0.72rem;font-weight:600'>{label}</span>")


def color_dot(color: str, size: int = 10) -> str:
    return (f"<span style='display:inline-block;width:{size}px;height:{size}px;border-radius:50%;"
            f"background:{color};margin-right:6px;vertical-align:middle'></span>")


def roster(include_optional: bool) -> list[str]:
    return list(config.CORE_METALS) + (list(config.OPTIONAL_METALS) if include_optional else [])


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
def init_state() -> None:
    ss = st.session_state
    ss.setdefault("refresh_nonce", 0)
    ss.setdefault("view_cfg", dict(DEFAULT_VIEW))
    if "active_metal" not in ss:
        # Deep link / browser reload: ?metal=gold restores the last selection
        qp = st.query_params.get("metal")
        ss.active_metal = qp if qp in METALS else None
    ss.setdefault("include_aluminum", ss.active_metal in config.OPTIONAL_METALS)


def set_selection(metal_key: str, view_cfg: dict[str, Any]) -> None:
    st.session_state.active_metal = metal_key
    st.session_state.view_cfg = view_cfg
    st.query_params["metal"] = metal_key


def clear_selection() -> None:
    st.session_state.active_metal = None
    st.query_params.pop("metal", None)


init_state()


# --------------------------------------------------------------------------- #
# Pop-up: choose metal + view settings
# --------------------------------------------------------------------------- #
@st.dialog("Choose a metal", width="large")
def choose_metal_dialog(preselect: str | None = None) -> None:
    """Runs as its own fragment: widget changes here rerun only the dialog."""
    ss = st.session_state
    options = roster(ss.include_aluminum)
    current = preselect or ss.active_metal
    current = current if current in options else options[0]
    cfg = ss.view_cfg

    picked = st.pills(
        "Metal", options, default=current,
        format_func=lambda k: METALS[k].name, key="dlg_metal",
    )
    metal = picked or current          # clicking the active pill again deselects it
    spec = METALS[metal]

    # Preview: pulls only the highlighted metal (cached, so browsing is cheap)
    with st.spinner(f"Loading {spec.name}…"):
        hist = fetch_history(metal, ss.get("lookback_days", config.HISTORY_LOOKBACK_DAYS), ss.refresh_nonce)
    last = float(hist["series"].iloc[-1])
    with st.container(border=True):
        st.markdown(
            f"{color_dot(spec.color, 12)}**{spec.name}** &nbsp;{status_badge(hist['status'])}<br>"
            f"<span style='font-size:0.85rem'>{spec.benchmark} · {spec.venue} · "
            f"quoted in {spec.native_unit}</span>",
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        c1.metric(f"Last ({hist['last_obs_date']})", fmt_price(last, spec.native_unit))
        c2.metric("Equivalent", fmt_price(convert_unit(last, spec.native_unit, config.Unit.USD_KG),
                                          config.Unit.USD_KG))
        if hist["status"] != STATUS_LIVE:
            st.caption(f"{hist['source']}. {hist['message']}")

    st.markdown("**View settings**")
    c1, c2 = st.columns(2)
    mode = c1.radio("Display units", DisplayMode.ALL, index=DisplayMode.ALL.index(cfg["mode"]))
    band = c2.radio("Risk band", list(BAND_OPTIONS), index=list(BAND_OPTIONS).index(cfg["band"]))
    c3, c4 = st.columns(2)
    horizon_days = c3.slider("Forecast days shown", 5, config.FORECAST_HORIZON, int(cfg["horizon_days"]))
    history_points = c4.slider("History shown (business days)", 20, 250, int(cfg["history_points"]), step=10)

    st.divider()
    b1, b2, _ = st.columns([1, 1, 2])
    if b1.button("Confirm", type="primary", width="stretch"):
        set_selection(metal, {"mode": mode, "band": band,
                              "horizon_days": horizon_days, "history_points": history_points})
        st.rerun()                     # closes the dialog, full rerun with new focus
    if b2.button("Cancel", width="stretch"):
        st.rerun()                     # closes without touching saved state


# --------------------------------------------------------------------------- #
# Sidebar (global data settings)
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Data & model")
    if st.button("🔄 Refresh Live Data", width="stretch",
                 help="Clears cached prices and re-pulls from the API."):
        clear_price_cache()
        st.session_state.refresh_nonce += 1
        st.toast("Price cache cleared. Pulling fresh data…")

    st.toggle("Include Aluminum in roster", key="include_aluminum",
              help="Optional metal; appears in the Choose Metal pop-up when on.")
    st.select_slider("History requested (calendar days)", options=[365, 545, 730, 1095],
                     value=config.HISTORY_LOOKBACK_DAYS, key="lookback_days")

    engine = get_forecast_engine()
    st.divider()
    st.caption("Forecast engine")
    if engine.client.available:
        st.success(f"{ENGINE_TIMESTRAL} backend configured", icon="✅")
    else:
        st.warning("Timestral backend not configured; random-walk baseline in use. "
                   "Set TIMESTRAL_API_URL in secrets.", icon="⚠️")
    st.caption(f"Horizon: {config.FORECAST_HORIZON} business days · Quantiles: "
               + ", ".join(config.quantile_label(q).upper() for q in config.QUANTILES))

# Keep selection valid if Aluminum was switched off while active
if st.session_state.active_metal and st.session_state.active_metal not in roster(st.session_state.include_aluminum):
    clear_selection()
    st.toast("Aluminum removed from roster; choose another metal.")

nonce = st.session_state.refresh_nonce
lookback_days = st.session_state.lookback_days


# --------------------------------------------------------------------------- #
# Landing: no metal chosen yet
# --------------------------------------------------------------------------- #
def render_landing() -> None:
    st.title("Metals 30-Day Probabilistic Forecast")
    st.markdown("Zero-shot quantile forecasts (P10 floor · P50 expected · P90 ceiling) "
                "from Timestral-28M. Pick a metal to see its outlook, risk band and calibration.")
    if st.button("Choose Metal", type="primary", icon="🪙", key="landing_choose"):
        choose_metal_dialog()

    st.markdown("#### Roster")
    keys = roster(st.session_state.include_aluminum)
    cols = st.columns(4)
    for i, key in enumerate(keys):
        spec = METALS[key]
        with cols[i % 4]:
            with st.container(border=True):
                st.markdown(f"{color_dot(spec.color, 12)}**{spec.name}**<br>"
                            f"<span style='font-size:0.8rem'>{spec.benchmark}<br>{spec.native_unit}</span>",
                            unsafe_allow_html=True)
                if st.button("Select", key=f"landing_pick_{key}", width="stretch"):
                    choose_metal_dialog(preselect=key)


# --------------------------------------------------------------------------- #
# Chart builders (consume the build_chart_payload() structure only)
# --------------------------------------------------------------------------- #
def _tick_format(payload: dict) -> str:
    """Pick axis precision from the displayed magnitude (e.g. zinc is ~3 USD/kg)."""
    last_vals = [s["history"][-1]["value"] for s in payload["series"] if s["axis"] == "left" and s["history"]]
    smallest = min(last_vals) if last_vals else 1000
    return ",.2f" if smallest < 50 else ",.1f" if smallest < 1000 else ",.0f"


def build_figure(payload: dict) -> go.Figure:
    fig = go.Figure()
    band = payload["band"]
    for s in payload["series"]:
        yaxis = "y2" if s["axis"] == "right" else "y"
        color, unit = s["color"], s["unit"]
        hx = [p["date"] for p in s["history"]]
        hy = [p["value"] for p in s["history"]]
        fx = [p["date"] for p in s["forecast"]]

        fig.add_trace(go.Scatter(
            x=hx, y=hy, mode="lines", name=f"{s['label']} actual", legendgroup=s["id"], yaxis=yaxis,
            line=dict(color=color, width=1.6),
            hovertemplate=f"<b>{s['label']}</b> actual<br>%{{x}}<br>%{{y:,.2f}} {unit}<extra></extra>",
        ))
        if band:
            lo, hi = band["lower"], band["upper"]
            fig.add_trace(go.Scatter(
                x=fx, y=[p[hi] for p in s["forecast"]], mode="lines", line=dict(width=0),
                legendgroup=s["id"], showlegend=False, hoverinfo="skip", yaxis=yaxis,
            ))
            fig.add_trace(go.Scatter(
                x=fx, y=[p[lo] for p in s["forecast"]], mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor=hex_to_rgba(color, 0.18),
                name=f"{lo.upper()}–{hi.upper()} band", legendgroup=s["id"], hoverinfo="skip", yaxis=yaxis,
            ))
        custom = np.array([[p["p10"], p["p90"]] for p in s["forecast"]])
        fig.add_trace(go.Scatter(
            x=[hx[-1]] + fx, y=[hy[-1]] + [p["p50"] for p in s["forecast"]],
            customdata=np.vstack([[hy[-1], hy[-1]], custom]),
            mode="lines", name="P50 forecast", legendgroup=s["id"], yaxis=yaxis,
            line=dict(color=color, width=2.4, dash="dash"),
            hovertemplate=(f"<b>{s['label']}</b> forecast<br>%{{x}}<br>"
                           f"P50 %{{y:,.2f}} {unit}<br>P10 %{{customdata[0]:,.2f}} · "
                           f"P90 %{{customdata[1]:,.2f}}<extra></extra>"),
        ))

    if payload["forecastStart"]:
        fig.add_vline(x=payload["forecastStart"], line_width=1, line_dash="dot", line_color="gray")
        fig.add_annotation(x=payload["forecastStart"], y=1.02, yref="paper", text="forecast start",
                           showarrow=False, font=dict(size=11, color="gray"))

    mode = payload["displayMode"]
    fig.update_layout(
        height=480, hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0),
        yaxis=dict(title=payload["axes"]["left"],
                   tickformat=_tick_format(payload)),
    )
    if payload["axes"].get("right"):
        fig.update_layout(yaxis2=dict(title=payload["axes"]["right"], overlaying="y", side="right",
                                      showgrid=False, tickformat=",.0f"))
    return fig


def calibration_figure(bt: dict, color: str) -> go.Figure:
    cal = bt["calibration"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Perfect calibration",
                             line=dict(color="gray", dash="dot", width=1)))
    fig.add_trace(go.Scatter(
        x=[c["nominal"] for c in cal], y=[c["observed"] for c in cal], mode="lines+markers+text",
        text=[c["label"] for c in cal], textposition="top left", name="Observed",
        line=dict(color=color, width=2), marker=dict(size=8),
        hovertemplate="%{text}: nominal %{x:.0f}% · observed %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(title="Nominal quantile (%)", range=[0, 100]),
                      yaxis=dict(title="Realized ≤ quantile (%)", range=[0, 100]),
                      legend=dict(orientation="h", y=-0.25))
    return fig


# --------------------------------------------------------------------------- #
# Focused view for the selected metal
# --------------------------------------------------------------------------- #
def render_metal_view(key: str, cfg: dict[str, Any]) -> None:
    spec = METALS[key]

    # Data + inference for this metal only (all cached)
    with st.spinner(f"Loading {spec.name} and running the forecast…"):
        hist = fetch_history(key, lookback_days, nonce)
        s = hist["series"]
        ctx = tuple(np.round(s.values[-config.CONTEXT_LENGTH:], 6).tolist())
        fc = forecast_metal(key, ctx, s.index[-1].date().isoformat(), refresh_nonce=nonce)
        bt = backtest_metal(key, tuple(np.round(s.values, 6).tolist()), refresh_nonce=nonce)
    histories, forecasts = {key: hist}, {key: fc}
    h_idx = int(cfg["horizon_days"]) - 1

    # ---- Header: active metal + switch button -------------------------------
    head, btn = st.columns([5, 1], vertical_alignment="center")
    head.markdown(
        f"<h2 style='margin:0'>{color_dot(spec.color, 16)}{spec.name} "
        f"<span style='font-size:0.9rem;font-weight:400'>30-day outlook</span></h2>",
        unsafe_allow_html=True,
    )
    if btn.button("Change metal", icon="🔁", type="primary", width="stretch", key="header_choose"):
        choose_metal_dialog()
    st.caption(f"View: {cfg['mode']} · {cfg['band']} · {cfg['horizon_days']} forecast days · "
               f"{cfg['history_points']} days of history")

    # ---- Foundation: provenance ---------------------------------------------
    st.markdown(
        f"<div style='border:1px solid rgba(128,128,128,0.35);border-radius:8px;padding:8px 12px;"
        f"font-size:0.85rem;line-height:1.8'>"
        f"<b>Data provenance</b> {status_badge(hist['status'])} · {hist['source']} · "
        f"as of {hist['as_of'].replace('T', ' ')} UTC · last observation {hist['last_obs_date']} · "
        f"engine <b>{fc['engine']}</b></div>",
        unsafe_allow_html=True,
    )
    if hist["status"] != STATUS_LIVE:
        st.caption(hist["message"])
    if fc["engine_note"] and fc["engine_note"] != "OK":
        st.caption(fc["engine_note"])
    if hist["status"] == STATUS_REFERENCE:
        st.info("This metal is on a synthetic REFERENCE series (no live data available). "
                "Figures show pipeline behavior only; do not use them for decisions.", icon="ℹ️")

    # ---- Foundation: KPI cards ----------------------------------------------
    st.subheader("Foundation")
    start = fc["start_value"]
    q = {lbl: fc["quantiles"][lbl][h_idx] for lbl in ("p10", "p50", "p90")}
    unit = spec.native_unit
    num = (lambda v: f"{v:,.0f}") if start >= 1000 else (lambda v: f"{v:,.2f}")
    st.caption(f"Prices in {unit} · forecast day D{cfg['horizon_days']} · start {fc['start_date']}")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Last price", num(start), help=f"{unit}, {spec.benchmark}", border=True)
    k2.metric(f"P50 expected · D{cfg['horizon_days']}", num(q["p50"]),
              delta=f"{pct(start, q['p50']):+.1f}%", border=True)
    k3.metric("P10 floor", num(q["p10"]), delta=f"{pct(start, q['p10']):+.1f}%", border=True)
    k4.metric("P90 ceiling", num(q["p90"]), delta=f"{pct(start, q['p90']):+.1f}%", border=True)
    k5.metric("P10–P90 width", f"{(q['p90'] - q['p10']) / start * 100:.1f}%",
              help="Spread of the 80% band as % of the start price. Wider = more uncertainty.", border=True)

    with st.expander("Benchmark & unit context"):
        usd_kg = convert_unit(start, unit, config.Unit.USD_KG)
        st.dataframe(pd.DataFrame([
            {"Field": "Benchmark", "Value": spec.benchmark},
            {"Field": "Venue", "Value": spec.venue},
            {"Field": "Native quote unit", "Value": unit},
            {"Field": "Last price (USD/kg)", "Value": f"{usd_kg:,.4f}"},
            {"Field": "API symbol", "Value": spec.api_symbol},
            {"Field": "Chart color", "Value": spec.color},
        ]), hide_index=True, width="stretch")
        st.caption("1 troy oz = 0.0311035 kg · 1 MT = 1,000 kg.")

    # ---- Forecast chart (from the Bklit-compatible payload) ------------------
    st.subheader("30-day forecast")
    payload = build_chart_payload(
        histories, forecasts, [key], cfg["mode"],
        int(cfg["history_points"]), int(cfg["horizon_days"]), BAND_OPTIONS[cfg["band"]],
    )
    st.plotly_chart(build_figure(payload), width="stretch", key=f"chart_{key}")

    with st.expander("Structured chart data (JSON props for Bklit UI / React)"):
        st.download_button("Download chart payload (.json)", json.dumps(payload, indent=2),
                           file_name=f"{key}_chart_payload.json", mime="application/json")
        st.json(payload, expanded=False)

    # ---- Risk calibration ----------------------------------------------------
    st.subheader("Risk & calibration")
    left, right = st.columns([2, 3])
    with left:
        if bt.get("available"):
            st.metric("MAPE (P50)", f"{bt['mape']:.1f}%", border=True)
            hit = bt["directional_hit_rate"]
            st.metric("Directional hit rate", f"{hit:.0f}%" if hit is not None else "n/a", border=True,
                      help="Share of forecast days where P50 called the up/down move correctly. "
                           "n/a when the P50 path is flat (e.g. the zero-drift baseline).")
            st.metric(f"{bt['band']} coverage", f"{bt['coverage']:.0f}%",
                      delta=f"{bt['coverage'] - bt['nominal_coverage']:+.0f} pts vs {bt['nominal_coverage']:.0f}% target",
                      delta_color="off", border=True)
            st.caption(f"Rolling-origin backtest: {bt['n_windows']} windows × {config.FORECAST_HORIZON} "
                       f"days ({bt['n_points']} points) · engine {', '.join(bt['engines'])}")
        else:
            st.info(bt.get("reason", "Backtest unavailable"))
    with right:
        if bt.get("available"):
            st.plotly_chart(calibration_figure(bt, spec.color), width="stretch", key=f"cal_{key}")

    table = pd.DataFrame(fc["quantiles"], index=pd.Index(fc["dates"], name="date"))
    table.columns = [c.upper() for c in table.columns]
    table["P50 Δ%"] = (table["P50"] / start - 1) * 100
    table["P10–P90 width %"] = (table["P90"] - table["P10"]) / start * 100
    with st.expander("Forecast quantile table"):
        st.dataframe(table.style.format("{:,.2f}"), width="stretch")
        d1, d2 = st.columns(2)
        d1.download_button(f"Download {spec.name} forecast (.csv)", table.to_csv(),
                           file_name=f"{key}_forecast_{fc['start_date']}.csv", mime="text/csv")
        d2.download_button(
            "Download provenance (.json)",
            json.dumps({"provenance": provenance_snapshot(histories), "forecast": fc,
                        "backtest": bt, "view": cfg}, indent=2, default=str),
            file_name=f"{key}_provenance.json", mime="application/json",
        )


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
with st.sidebar:
    if st.session_state.active_metal:
        st.divider()
        if st.button("Choose another metal", width="stretch", key="sidebar_choose"):
            choose_metal_dialog()
        if st.button("Back to roster", width="stretch", key="sidebar_reset"):
            clear_selection()
            st.rerun()

if st.session_state.active_metal:
    render_metal_view(st.session_state.active_metal, st.session_state.view_cfg)
else:
    render_landing()

st.caption("Forecasts are probabilistic model outputs, not investment advice. "
           "Quantile bands describe model uncertainty and can under-state tail risk.")
