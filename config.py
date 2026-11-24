"""Central configuration: metal metadata, units, colors, forecast settings, secrets.

Everything that describes *what* a metal is (benchmark, venue, quote unit, API
symbol, chart color) lives here so the data, model and UI layers never
hardcode metal-specific details.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import streamlit as st

# --------------------------------------------------------------------------- #
# Units
# --------------------------------------------------------------------------- #
TROY_OZ_KG = 0.0311034768
METRIC_TON_KG = 1000.0


class Unit:
    USD_MT = "USD/MT"
    USD_OZT = "USD/troy oz"
    USD_KG = "USD/kg"


# kilograms contained in one quote unit (price-per-unit -> price-per-kg)
KG_PER_UNIT: dict[str, float] = {
    Unit.USD_MT: METRIC_TON_KG,
    Unit.USD_OZT: TROY_OZ_KG,
    Unit.USD_KG: 1.0,
}


class DisplayMode:
    INDEXED = "Indexed (Base-100 at forecast start)"
    NATIVE = "Native quote units"
    USD_KG = "Normalized USD/kg"

    ALL = (INDEXED, NATIVE, USD_KG)


# --------------------------------------------------------------------------- #
# Metal metadata
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MetalSpec:
    key: str
    name: str
    benchmark: str            # human-readable benchmark description
    venue: str                # LME / LBMA / LPPM / OTC
    native_unit: str          # Unit.USD_MT or Unit.USD_OZT
    color: str                # fixed hex color, never derived from list order
    api_symbol: str           # symbol on the configured price API
    api_unit: str             # unit the API returns for this symbol
    api_inverted: bool        # True if API returns "metal per 1 USD"
    reference_anchor: float   # illustrative level for synthetic reference series
    reference_daily_vol: float
    optional: bool = False    # optional metals are off unless toggled on


# NOTE on API symbols/units: the values below follow the Metals-API style
# ("XAU", "XPT", "XRH", "NI", ...). Check them against the symbol list and unit
# conventions of *your* provider and plan before going live; base metals in
# particular are often returned per troy oz or per lb rather than per tonne.
# `api_unit` handles the conversion into `native_unit` automatically.
#
# `reference_anchor` values are illustrative levels for the offline synthetic
# series only. They are NOT market quotes.
METALS: dict[str, MetalSpec] = {
    "nickel": MetalSpec(
        key="nickel", name="Nickel", benchmark="LME Nickel Official Cash",
        venue="LME", native_unit=Unit.USD_MT, color="#1F77B4",
        api_symbol="NI", api_unit=Unit.USD_MT, api_inverted=True,
        reference_anchor=16_000.0, reference_daily_vol=0.020,
    ),
    "platinum": MetalSpec(
        key="platinum", name="Platinum", benchmark="LBMA/LPPM Platinum Price (spot)",
        venue="LBMA / LPPM", native_unit=Unit.USD_OZT, color="#6C5CE7",
        api_symbol="XPT", api_unit=Unit.USD_OZT, api_inverted=True,
        reference_anchor=1_000.0, reference_daily_vol=0.017,
    ),
    "tin": MetalSpec(
        key="tin", name="Tin", benchmark="LME Tin Official Cash",
        venue="LME", native_unit=Unit.USD_MT, color="#17A589",
        api_symbol="TIN", api_unit=Unit.USD_MT, api_inverted=True,
        reference_anchor=30_000.0, reference_daily_vol=0.018,
    ),
    "zinc": MetalSpec(
        key="zinc", name="Zinc", benchmark="LME Zinc Official Cash",
        venue="LME", native_unit=Unit.USD_MT, color="#E67E22",
        api_symbol="ZNC", api_unit=Unit.USD_MT, api_inverted=True,
        reference_anchor=2_800.0, reference_daily_vol=0.015,
    ),
    "lead": MetalSpec(
        key="lead", name="Lead", benchmark="LME Lead Official Cash",
        venue="LME", native_unit=Unit.USD_MT, color="#5D6D7E",
        api_symbol="LEAD", api_unit=Unit.USD_MT, api_inverted=True,
        reference_anchor=2_000.0, reference_daily_vol=0.014,
    ),
    "rhodium": MetalSpec(
        key="rhodium", name="Rhodium", benchmark="OTC dealer reference (spot)",
        venue="OTC / dealer", native_unit=Unit.USD_OZT, color="#C0392B",
        api_symbol="XRH", api_unit=Unit.USD_OZT, api_inverted=True,
        reference_anchor=5_000.0, reference_daily_vol=0.025,
    ),
    "gold": MetalSpec(
        key="gold", name="Gold", benchmark="LBMA Gold Price (spot)",
        venue="LBMA", native_unit=Unit.USD_OZT, color="#D4AC0D",
        api_symbol="XAU", api_unit=Unit.USD_OZT, api_inverted=True,
        reference_anchor=2_500.0, reference_daily_vol=0.009,
    ),
    "aluminum": MetalSpec(
        key="aluminum", name="Aluminum", benchmark="LME Aluminium Official Cash",
        venue="LME", native_unit=Unit.USD_MT, color="#7DCEA0",
        api_symbol="ALU", api_unit=Unit.USD_MT, api_inverted=True,
        reference_anchor=2_500.0, reference_daily_vol=0.013,
        optional=True,
    ),
}

CORE_METALS: tuple[str, ...] = tuple(k for k, m in METALS.items() if not m.optional)
OPTIONAL_METALS: tuple[str, ...] = tuple(k for k, m in METALS.items() if m.optional)

# --------------------------------------------------------------------------- #
# Forecast / data settings
# --------------------------------------------------------------------------- #
FORECAST_HORIZON = 30                 # steps
FORECAST_FREQ = "B"                   # business days; metals don't trade weekends
QUANTILES: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)
CONTEXT_LENGTH = 512                  # max history points sent to the model
HISTORY_LOOKBACK_DAYS = 730           # calendar days requested from the API
BACKTEST_ORIGINS = 4                  # rolling-origin evaluation windows
MIN_CONTEXT_POINTS = 60

CACHE_TTL_SECONDS = 3600
HTTP_TIMEOUT_SECONDS = 20
API_MAX_SPAN_DAYS = 365               # max days per timeseries request on most plans
DISK_CACHE_DIR = ".data_cache"        # last good live pull per metal

TIMESTRAL_MODEL_NAME = "timestral-28m"


def quantile_label(q: float) -> str:
    return f"p{int(round(q * 100))}"


# --------------------------------------------------------------------------- #
# Secrets
# --------------------------------------------------------------------------- #
def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read from st.secrets first, then environment variables. Never hardcoded."""
    try:
        value = st.secrets.get(name)  # raises if no secrets.toml exists
        if value not in (None, ""):
            return str(value)
    except Exception:
        pass
    value = os.getenv(name)
    return value if value not in (None, "") else default


def metals_api_settings() -> dict[str, Optional[str]]:
    return {
        "base_url": get_secret("METALS_API_BASE_URL", "https://metals-api.com/api"),
        "api_key": get_secret("METALS_API_KEY"),
    }


def timestral_settings() -> dict[str, Optional[str]]:
    return {
        "url": get_secret("TIMESTRAL_API_URL"),
        "api_key": get_secret("TIMESTRAL_API_KEY"),
        "model": get_secret("TIMESTRAL_MODEL", TIMESTRAL_MODEL_NAME),
    }
