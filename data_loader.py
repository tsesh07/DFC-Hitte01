"""
data_loader.py — Single source of truth for loading the Arduino sensor data.

Reads ../data/raw/DATA.CSV (the SD-card export, never edited) and returns a
clean dataframe ready for the dashboard. All transformations live here so the
pipeline is reproducible end-to-end from the original SD card export.
"""
from pathlib import Path
import pandas as pd
import streamlit as st

# Resolve raw data path relative to this file — works regardless of cwd or deploy env
APP_DIR = Path(__file__).parent
RAW_PATH = APP_DIR / "data" / "raw" / "DATA.CSV"

# Each walk is identified by its date prefix in the timestamp column
SESSIONS = {
    "2026-05-06": "Day 1 - lid closed",
    "2026-05-08": "Day 2 - lid open",
}

# Walk windows (gps_time as HHMMSS) — crop out sensor warmup/idle outside the walk.
# Set either bound to None to keep all rows for that session.
WALK_WINDOWS = {
    "Day 1 - lid closed": (121155, 130545),
    "Day 2 - lid open":   (None, None),
}


def _nmea_to_decimal(coord, hemi):
    """NMEA DDDMM.MMMM + hemisphere letter -> decimal degrees."""
    if pd.isna(coord) or pd.isna(hemi):
        return None
    deg = int(coord // 100)
    minutes = coord - deg * 100
    dec = deg + minutes / 60
    return -dec if hemi in ("S", "W") else dec


def _repair_corrupt_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Repair RTC-glitch timestamps ('2001-51-14 ...') using gps_time.

    These rows fall inside the Day 2 walk, so we reconstruct them as
    2026-05-08 + HH:MM:SS pulled from gps_time. The gps_time column comes
    directly from the GPS module and is unaffected by the RTC glitch.
    """
    bad = df["timestamp"].astype(str).str.startswith("2001")
    if not bad.any():
        return df

    def _build(g):
        if pd.isna(g):
            return None
        g = int(g)
        return f"2026-05-08 {g//10000:02d}:{(g//100)%100:02d}:{g%100:02d}"

    df.loc[bad, "timestamp"] = df.loc[bad, "gps_time"].apply(_build)
    return df


@st.cache_data
def load_data(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load and clean the raw SD-card export.

    Pipeline:
      1. Read raw CSV
      2. Drop fully empty rows (Excel-saved file artifacts)
      3. Repair RTC-glitch timestamps from gps_time
      4. Parse timestamp -> datetime; drop any unparseable rows
      5. Filter to our two walk dates (removes other students' April data)
      6. Add session label column
      7. Crop each session to its actual walk window via gps_time
      8. Convert NMEA lat/lon to decimal degrees
    """
    df = pd.read_csv(path)
    df = df.dropna(how="all").reset_index(drop=True)
    df = _repair_corrupt_timestamps(df)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()

    df["date_str"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    df = df[df["date_str"].isin(SESSIONS)].copy()
    df["session"] = df["date_str"].map(SESSIONS)

    keep = pd.Series(False, index=df.index)
    for session, (start, end) in WALK_WINDOWS.items():
        m = df["session"] == session
        if start is not None:
            m &= df["gps_time"] >= start
        if end is not None:
            m &= df["gps_time"] <= end
        keep |= m
    df = df[keep].copy()

    df["lat_dec"] = df.apply(lambda r: _nmea_to_decimal(r["lat"], r["ns"]), axis=1)
    df["lon_dec"] = df.apply(lambda r: _nmea_to_decimal(r["lon"], r["ew"]), axis=1)

    return df.drop(columns=["date_str"]).reset_index(drop=True)