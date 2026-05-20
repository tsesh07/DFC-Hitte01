"""
Arduino Sensor Logger Dashboard
================================
Leest de schoongemaakte multi-sessie DataFrame via data_loader.load_data() en
visualiseert de GPS-route + omgevingssensor-metingen, met ondersteuning voor
het vergelijken van Dag 1 (deksel dicht) vs Dag 2 (deksel open).

Starten met:  streamlit run app.py
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import linemerge, polygonize, unary_union
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Statistische toetsen
from scipy import stats as scipy_stats


# Voor KNMI-API requests en OSM Overpass
import urllib.request
import urllib.parse
import urllib.error
import io

from data_loader import load_data, SESSION_METADATA, GPS_ONLY_SESSIONS

# --------------------------------------------------------------------------
# Pagina-configuratie
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Sensor Logger Dashboard",
    page_icon="📡",
    layout="wide",
)

# --------------------------------------------------------------------------
# Custom CSS — Urban Dark theme
# --------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Layout ────────────────────────────────────────────────────────────── */
.main .block-container { padding: 1.5rem 3rem 3rem; max-width: 1400px; }

/* ── Sidebar ────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] { border-right: 1px solid #334155; }
section[data-testid="stSidebar"] hr { border-color: #334155 !important; opacity:1; }

/* ── Tabs ───────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: #1e293b;
    border-radius: 10px;
    padding: 5px 6px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 7px !important;
    color: #94a3b8 !important;
    font-weight: 500;
    padding: 8px 18px !important;
    transition: color 0.15s !important;
    border: none !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #e2e8f0 !important; }
.stTabs [aria-selected="true"] {
    background: #38bdf8 !important;
    color: #0f172a !important;
    font-weight: 700 !important;
}

/* ── Metric cards ───────────────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.25);
    transition: transform 0.15s, box-shadow 0.15s;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(56,189,248,0.15);
}
[data-testid="stMetricValue"] { color: #38bdf8 !important; }
[data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: .85rem !important; }
[data-testid="stMetricDelta"] { color: #64748b !important; }

/* ── Buttons ────────────────────────────────────────────────────────────── */
.stButton > button, .stDownloadButton > button {
    background: linear-gradient(135deg, #0369a1, #0891b2) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: .3px;
    transition: all 0.2s ease !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: linear-gradient(135deg, #0284c7, #06b6d4) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(56,189,248,0.35) !important;
}

/* ── Expanders ──────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
}

/* ── DataFrames ─────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid #334155 !important;
    border-radius: 10px;
    overflow: hidden;
}

/* ── Dividers ───────────────────────────────────────────────────────────── */
hr { border-color: #334155 !important; opacity: 1; }

/* ── Insight cards (custom HTML components) ─────────────────────────────── */
.insight-card {
    background: #1e293b;
    border-radius: 12px;
    padding: 1.35rem 1.5rem;
    border-left: 4px solid #38bdf8;
    height: 185px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    overflow: hidden;
}
.insight-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.45);
}
.insight-card--blue  { border-left-color: #38bdf8; }
.insight-card--amber { border-left-color: #f59e0b; }
.insight-card--green { border-left-color: #10b981; }
.insight-card .card-title {
    color: #f1f5f9;
    font-size: 1rem;
    font-weight: 700;
    margin: 0 0 .55rem;
}
.insight-card .card-body {
    color: #94a3b8;
    font-size: .875rem;
    line-height: 1.65;
    margin: 0;
}
.insight-card .card-body strong { color: #e2e8f0; }

/* ── Hero banner ────────────────────────────────────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg, #1e3a5f 0%, #0f4c75 45%, #0d9488 100%);
    border-radius: 16px;
    padding: 2.25rem 2.75rem;
    margin-bottom: 1.75rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.45);
    border: 1px solid #1e4a6e;
}
.hero-banner h1 {
    color: #fff !important;
    font-size: 1.95rem !important;
    font-weight: 800 !important;
    margin: 0 0 .4rem !important;
    letter-spacing: -.5px;
}
.hero-banner .hero-sub {
    color: rgba(255,255,255,.8);
    font-size: .98rem;
    margin: 0 0 .25rem;
}
.hero-banner .hero-meta {
    color: rgba(255,255,255,.6);
    font-size: .84rem;
    margin: 0;
}

/* ── Section header accent ──────────────────────────────────────────────── */
.section-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: #f1f5f9;
    padding-bottom: .45rem;
    border-bottom: 2px solid #334155;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Coördinatensystemen
# --------------------------------------------------------------------------
# WGS84 (EPSG:4326)   — standaard "GPS" CRS, lat/lon in graden.
#                       Wat de Arduino's GPS-module rapporteert.
# RD New (EPSG:28992) — Nederlands rijksdriehoekstelsel, coördinaten in meters.
#                       Gebruikt voor operaties met echte afstanden (buffers,
#                       'straal in meters' berekeningen, enz.).
WGS84 = "EPSG:4326"
RD_NEW = "EPSG:28992"

ZONE_COLOURS = {
    "Museumplein":     "#ef4444",
    "Frans Halsbuurt": "#3b82f6",
    "Sarphatipark":    "#10b981",
    "Onderweg":        "#9ca3af",
}

# Onderscheidende, toegankelijke kleuren voor sessie-vergelijking
SESSION_COLOURS = {
    "Dag 1 - deksel dicht": "#2563eb",
    "Dag 2 - deksel open":  "#f97316",
    "Dag 3 - GPS track":    "#10b981",
    "Dag 4 - deksel dicht": "#a855f7",
}

# --------------------------------------------------------------------------
# Kaartstijlen voor Scattermap
# --------------------------------------------------------------------------
# Plotly's Scattermap (MapLibre) accepteert óf een ingebouwde stijlnaam,
# óf een volledige MapLibre style-spec als dict. We gebruiken een dict om
# de PDOK luchtfoto als raster-bron toe te voegen — landsdekkend, 25cm
# resolutie, kosteloos, en geen API-key nodig.
#
# Endpoint geverifieerd via PDOK GetCapabilities:
# https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0?request=GetCapabilities&service=wmts
PDOK_LUCHTFOTO_STYLE = {
    "version": 8,
    "sources": {
        "pdok-luchtfoto": {
            "type": "raster",
            "tiles": [
                "https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0/"
                "Actueel_ortho25/EPSG:3857/{z}/{x}/{y}.jpeg"
            ],
            "tileSize": 256,
            "attribution": "© PDOK / Beeldmateriaal Nederland",
        }
    },
    "layers": [
        {
            "id": "pdok-luchtfoto-layer",
            "type": "raster",
            "source": "pdok-luchtfoto",
            "minzoom": 0,
            "maxzoom": 22,
        }
    ],
}

# Mapping van leesbare Nederlandse labels naar Plotly map-styles.
# String-waarden zijn ingebouwde MapLibre-stijlen; dict-waarden zijn
# volledige style-specs die we zelf definiëren (zoals de PDOK luchtfoto).
MAP_STYLES = {
    "Straatkaart (OpenStreetMap)":   "open-street-map",
    "Lichte kaart (Carto Positron)": "carto-positron",
    "Donkere kaart (Carto Dark)":    "carto-darkmatter",
    "Luchtfoto (PDOK, 25cm NL)":     PDOK_LUCHTFOTO_STYLE,
}


# --------------------------------------------------------------------------
# Geo-hulpfuncties
# --------------------------------------------------------------------------
_ZONE_DIR = Path(__file__).parent / "data" / "zone"

# CBS buurtnaam → dashboard zone label
_BUURT_TO_ZONE = {
    "Museumplein":      "Museumplein",
    "Frans Halsbuurt":  "Frans Halsbuurt",
    "Sarphatiparkbuurt": "Sarphatipark",
}

# Bodemgebruik-bestanden per zone (klasse-oppervlakken in m²)
_LAND_COVER_FILES = {
    "Frans Halsbuurt": "FransHalsbuurt_opp.geojson",
    "Museumplein":     "Museumplein_opp.geojson",
    "Sarphatipark":    "Sarphatiparkbuurt_opp.geojson",
}


@st.cache_data(show_spinner=False)
def build_zones_gdf() -> gpd.GeoDataFrame:
    """Laad precieze CBS-buurtgrenzen uit het lokale GeoJSON-bestand."""
    gdf = gpd.read_file(_ZONE_DIR / "gekozen_wijken.geojson")
    gdf = gdf[gdf["buurtnaam"].isin(_BUURT_TO_ZONE)].copy()
    gdf["zone"]    = gdf["buurtnaam"].map(_BUURT_TO_ZONE)
    gdf["_source"] = "GeoJSON (CBS buurtgrenzen)"
    return gdf[["zone", "_source", "geometry"]].to_crs(RD_NEW).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_land_cover() -> dict[str, dict]:
    """Bereken bodemgebruik-percentages per zone, inclusief subcategorieën."""
    result = {}
    for zone, fname in _LAND_COVER_FILES.items():
        gdf = gpd.read_file(_ZONE_DIR / fname)

        # Hoofdklassen: verhard / groen / gebouw / water (% van totaal)
        totals = gdf.groupby("klasse")["opp"].sum()
        total  = totals.sum()
        klasse_pct = {k: round(float(v / total * 100), 1) for k, v in totals.items()}

        # Verharding detail: gesloten vs open (% binnen verhard)
        v_gdf   = gdf[gdf["klasse"] == "verhard"]
        v_subs  = v_gdf.groupby("fysiek_voorkomen")["opp"].sum()
        v_total = v_subs.sum()
        v_sub: dict[str, float] = {}
        for k, v in v_subs.items():
            label = k if k in ("gesloten verharding", "open verharding") else "overig"
            v_sub[label] = round(v_sub.get(label, 0.0) + float(v / v_total * 100), 1)

        # Groen detail: groenvoorziening (openbaar) vs erf (privé) (% binnen groen)
        g_gdf   = gdf[gdf["klasse"] == "groen"]
        g_subs  = g_gdf.groupby("fysiek_voorkomen")["opp"].sum()
        g_total = g_subs.sum()
        g_sub: dict[str, float] = {}
        for k, v in g_subs.items():
            label = k if k in ("groenvoorziening", "erf") else "overig"
            g_sub[label] = round(g_sub.get(label, 0.0) + float(v / g_total * 100), 1)

        result[zone] = {
            "klasse":         klasse_pct,
            "verharding_sub": v_sub,
            "groen_sub":      g_sub,
        }
    return result


def assign_zones_via_sjoin(df: pd.DataFrame, zones_gdf: gpd.GeoDataFrame) -> pd.Series:
    """Ruimtelijke join: voor elk monster zoeken we in welke zone-polygoon het valt."""
    valid = df["lat_dec"].notna() & df["lon_dec"].notna()

    points = gpd.GeoDataFrame(
        df.loc[valid, ["lat_dec", "lon_dec"]].copy(),
        geometry=gpd.points_from_xy(df.loc[valid, "lon_dec"], df.loc[valid, "lat_dec"]),
        crs=WGS84,
    ).to_crs(RD_NEW)

    joined = gpd.sjoin(points, zones_gdf[["zone", "geometry"]],
                       how="left", predicate="within")

    result = pd.Series("Onderweg", index=df.index, dtype="object")
    result.loc[valid] = joined["zone"].fillna("Onderweg").values
    return result


# --------------------------------------------------------------------------
# Route- / sensor-hulpfuncties
# --------------------------------------------------------------------------
def haversine_m(lat1, lon1, lat2, lon2):
    """Groot-cirkel afstand in meters tussen twee punten (vectorised)."""
    R = 6_371_000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def add_track_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Bereken stapafstand (m), cumulatieve afstand (m), en snelheid (m/s, km/h).

    Berekend PER SESSIE zodat er geen fictieve 'stap' ontstaat tussen het
    laatste punt van Dag 1 en het eerste punt van Dag 2.
    """
    pieces = []
    for _, sub in df.groupby("session", sort=False):
        sub = sub.copy()
        sub["step_m"] = haversine_m(
            sub["lat_dec"].shift(), sub["lon_dec"].shift(),
            sub["lat_dec"],         sub["lon_dec"],
        )
        dt = sub["timestamp"].diff().dt.total_seconds()
        sub["speed_ms"] = sub["step_m"] / dt.replace(0, np.nan)
        sub["speed_kmh"] = sub["speed_ms"] * 3.6
        sub["cum_dist_m"] = sub["step_m"].fillna(0).cumsum()
        pieces.append(sub)
    return pd.concat(pieces).sort_index()


def add_minutes_from_start(df: pd.DataFrame) -> pd.DataFrame:
    """Voeg `minute_from_start` toe: minuten sinds het eerste monster per sessie."""
    df = df.copy()
    df["minute_from_start"] = df.groupby("session")["timestamp"].transform(
        lambda x: (x - x.min()).dt.total_seconds() / 60
    )
    return df


def add_drift_correction(df: pd.DataFrame) -> pd.DataFrame:
    """Subtract per-session linear self-heating drift from tempC.

    The Arduino board warms up over the walk, producing a systematic upward
    trend in tempC unrelated to the urban environment. This fits a linear
    regression (tempC ~ minute_from_start) per session and subtracts the
    trend while preserving each session's mean temperature. The result,
    tempC_detrended, isolates the spatial microclimate signal from the
    instrument drift.
    """
    df = df.copy()
    df["tempC_detrended"] = np.nan
    for s, grp in df.groupby("session"):
        valid = grp["tempC"].notna() & grp["minute_from_start"].notna()
        if valid.sum() < 10:
            continue
        x = grp.loc[valid, "minute_from_start"].values
        y = grp.loc[valid, "tempC"].values
        slope, *_ = scipy_stats.linregress(x, y)
        mean_x = x.mean()
        df.loc[grp.index, "tempC_detrended"] = (
            grp["tempC"] - slope * (grp["minute_from_start"] - mean_x)
        )
    return df


def clean_sensor_data(df: pd.DataFrame, drop_glitches: bool) -> pd.DataFrame:
    """Maskeer optioneel duidelijke sensorstoringen."""
    df = df.copy()
    if drop_glitches:
        # BMP280 heeft ~1s opwarmtijd — drukmetingen ver onder zeeniveau zijn fout
        df.loc[df["pressure_hPa"] < 900, "pressure_hPa"] = np.nan
        # Sentinel-waarde van de fotosensor vóór de eerste echte meting
        df.loc[df["light_lux"] < 0, "light_lux"] = np.nan
        # Extreme tempC-waarden = sensor losgekoppeld.
        # 5 °C is een veilige ondergrens voor Amsterdam in de lente.
        # Dag 2 heeft ook hoge uitschieters (+144 °C spikes), boven ook clampen.
        df.loc[(df["tempC"] < 5) | (df["tempC"] > 50), "tempC"] = np.nan
    return df


# --------------------------------------------------------------------------
# KNMI weer-data fetcher (Schiphol, station 240)
# --------------------------------------------------------------------------
# KNMI biedt historische uurgegevens via een publieke POST-API. De data
# wordt tijdens kantooruren gevalideerd, dus zeer recente dagen (binnen
# 1-7 dagen) zijn vaak nog niet beschikbaar. We cachen succesvolle
# responses 24u; exceptions worden NIET gecached door st.cache_data, dus
# een mislukte fetch wordt automatisch opnieuw geprobeerd.
KNMI_ENDPOINT = "https://www.daggegevens.knmi.nl/klimatologie/uurgegevens"
KNMI_STATION_SCHIPHOL = 240

# --------------------------------------------------------------------------
# KNMI weer-data — de OSM/API zone-fetch is vervangen door lokale GeoJSONs


class KNMIFetchError(Exception):
    """Raised wanneer KNMI-data niet opgehaald kan worden, met diagnostische tekst."""


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_knmi_weather(date_str: str,
                      station: int = KNMI_STATION_SCHIPHOL) -> pd.DataFrame:
    """Haal uurlijkse KNMI-weerdata op voor één dag.

    date_str: 'YYYY-MM-DD'

    Returns een DataFrame met kolommen [hour_utc, temp_C, wind_ms,
    cloud_okta, radiation_jcm2], of raised KNMIFetchError met een
    beschrijvende foutmelding die we in de UI kunnen tonen.

    KNMI-eenheden: T in 0.1 °C, FH in 0.1 m/s, N in oktas (0-9),
    Q in J/cm². We converteren naar SI-eenheden.

    We gebruiken CSV-formaat (de KNMI-default) in plaats van JSON omdat
    JSON in de praktijk minder stabiel bleek voor zeer recente data.
    """
    ymd = date_str.replace("-", "")
    post_data = urllib.parse.urlencode({
        "stns":  str(station),
        "vars":  "T:FH:N:Q",
        "start": f"{ymd}01",
        "end":   f"{ymd}24",
        # geen fmt= → KNMI default = CSV (stabieler dan JSON)
    }).encode("utf-8")

    req = urllib.request.Request(
        KNMI_ENDPOINT,
        data=post_data,
        headers={"Content-Type": "application/x-www-form-urlencoded",
                 "User-Agent":   "microclimate-dashboard/1.0"},
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        raise KNMIFetchError(
            f"HTTP {e.code} {e.reason} — KNMI weigerde het verzoek."
        ) from e
    except urllib.error.URLError as e:
        raise KNMIFetchError(
            f"Netwerkfout: kon KNMI niet bereiken ({e.reason}). "
            "Check internet of firewall."
        ) from e
    except TimeoutError as e:
        raise KNMIFetchError("Timeout (>15s) — KNMI reageerde niet op tijd.") from e
    except Exception as e:
        raise KNMIFetchError(f"Onverwachte fout: {type(e).__name__}: {e}") from e

    if not raw.strip():
        raise KNMIFetchError(
            f"Lege response voor {date_str}. KNMI valideert uurgegevens "
            "tijdens kantooruren, dus zeer recente dagen kunnen ontbreken."
        )

    # KNMI CSV-respons: commentaarregels beginnen met '#', daarna komen de
    # data-regels. We negeren commentaar en parsen de rest.
    data_lines = [ln for ln in raw.splitlines()
                  if ln.strip() and not ln.lstrip().startswith("#")]
    if not data_lines:
        raise KNMIFetchError(
            f"Geen data-regels voor {date_str} in de KNMI-respons. Dit "
            "gebeurt bij data die nog niet gevalideerd is — uurgegevens "
            "hebben enkele dagen vertraging. Probeer over een paar dagen "
            "opnieuw, of voer de waarden handmatig in via de zijbalk."
        )

    try:
        df = pd.read_csv(
            io.StringIO("\n".join(data_lines)),
            header=None,
            names=["STN", "YYYYMMDD", "HH", "T", "FH", "N", "Q"],
            skipinitialspace=True,
            na_values=["", " ", "     "],
        )
    except Exception as e:
        raise KNMIFetchError(
            f"CSV-parsing mislukt: {type(e).__name__}: {e}. "
            f"Eerste 200 tekens van de respons: {raw[:200]!r}"
        ) from e

    if df.empty:
        raise KNMIFetchError(f"KNMI-data leeg na parsen voor {date_str}.")

    for col in ("HH", "T", "FH", "N", "Q"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return pd.DataFrame({
        "hour_utc":       df["HH"].astype("Int64"),
        "temp_C":         df["T"] / 10,
        "wind_ms":        df["FH"] / 10,
        "cloud_okta":     df["N"],
        "radiation_jcm2": df["Q"],
    }).dropna(subset=["hour_utc"])


def knmi_summary_for_walk(session_df: pd.DataFrame,
                          knmi_df: pd.DataFrame) -> dict:
    """Knip de KNMI-uurdata naar het wandelvenster en bereken samenvatting.

    Gebruikt gps_time (UTC) om te bepalen welke KNMI-uren overlappen.
    """
    if session_df.empty or knmi_df is None or knmi_df.empty:
        return {}
    gps_times = session_df["gps_time"].dropna().astype(int)
    if gps_times.empty:
        return {}
    start_hour = gps_times.min() // 10000
    end_hour   = gps_times.max() // 10000
    # KNMI uur 1 = data van 00:00-01:00 UTC, dus uur N dekt [N-1, N]
    overlap = knmi_df[(knmi_df["hour_utc"] >= start_hour) &
                      (knmi_df["hour_utc"] <= end_hour + 1)]
    if overlap.empty:
        return {}
    return {
        "temp_C":         overlap["temp_C"].mean(),
        "wind_ms":        overlap["wind_ms"].mean(),
        "cloud_okta":     overlap["cloud_okta"].mean(),
        "radiation_jcm2": overlap["radiation_jcm2"].mean(),
        "hours_covered":  f"{start_hour:02d}:00–{end_hour+1:02d}:00 UTC",
    }




# --------------------------------------------------------------------------
# Zijbalk — databron en filtering
# --------------------------------------------------------------------------
st.sidebar.header("⚙️ Data-instellingen")

try:
    raw = load_data()
except FileNotFoundError as e:
    st.error(f"Kon de ruwe data niet laden: {e}\n\n"
             "Verwacht op `data/raw/DATA.CSV` ten opzichte van `app.py`.")
    st.stop()

available_sessions = list(raw["session"].unique())

sessions = st.sidebar.multiselect(
    "Sessies om te tonen",
    options=available_sessions,
    default=available_sessions,
    help="Kies één voor een volledig single-session beeld, of beide om te vergelijken.",
)

if not sessions:
    st.warning("Selecteer minstens één sessie in de zijbalk.")
    st.stop()

drop_glitches = st.sidebar.toggle(
    "Sensorstoringen maskeren", value=True,
    help="Verberg BMP280-opwarming, −1 lux sentinels, en buiten-bereik "
         "temperatuurpieken (sensor-disconnects).",
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Geladen data**")
for s in available_sessions:
    n = (raw["session"] == s).sum()
    st.sidebar.markdown(f"- {s}: **{n:,}** rijen")

st.sidebar.markdown("---")
st.sidebar.caption(
    "**Dag 1**: schone data-acquisitie.  \n"
    "**Dag 2**: intermittente USB-stroomstoringen — voorzichtig interpreteren.  \n"
    "**Dag 3**: alleen GPS-track (BMP280 I2C-uitval).  \n"
    "**Dag 4**: schone replicatie van Dag 1 (deksel dicht, zelfde route)."
)

# --------------------------------------------------------------------------
# Handmatige KNMI-invoer (fallback wanneer auto-fetch faalt)
# --------------------------------------------------------------------------
with st.sidebar.expander("🌤️ KNMI handmatig instellen", expanded=False):
    st.caption(
        "Vul handmatige waarden in als de automatische KNMI-fetch faalt "
        "(recente data heeft doorgaans enkele dagen validatievertraging). "
        "Lege velden vallen terug op de auto-fetch."
    )
    use_manual = st.toggle("Handmatige waarden gebruiken", value=False)
    manual_knmi: dict[str, dict] = {}
    if use_manual:
        for s in available_sessions:
            st.markdown(f"**{s}**")
            c_a, c_b = st.columns(2)
            manual_knmi[s] = {
                "temp_C":         c_a.number_input(
                    "Temp °C", value=None, placeholder="auto",
                    key=f"mk_temp_{s}", step=0.1, format="%.1f"),
                "wind_ms":        c_b.number_input(
                    "Wind m/s", value=None, placeholder="auto",
                    key=f"mk_wind_{s}", step=0.1, format="%.1f"),
                "cloud_okta":     c_a.number_input(
                    "Bewolking (0-9)", value=None, placeholder="auto",
                    key=f"mk_cloud_{s}", step=1.0, min_value=0.0, max_value=9.0),
                "radiation_jcm2": c_b.number_input(
                    "Straling J/cm²", value=None, placeholder="auto",
                    key=f"mk_rad_{s}", step=10.0),
            }

# --------------------------------------------------------------------------
# Filter & verrijk de DataFrame
# --------------------------------------------------------------------------
df = raw[raw["session"].isin(sessions)].copy()
df = clean_sensor_data(df, drop_glitches)
df = add_track_metrics(df)
df = add_minutes_from_start(df)
df = add_drift_correction(df)

zones_gdf = build_zones_gdf()
df["zone"] = assign_zones_via_sjoin(df, zones_gdf)

# GPS-track sessies bevatten geen betrouwbare sensordata — apart houden zodat
# alle sensor-analyses uitsluitend op Dag 1 / Dag 2 draaien.
df_sensor = df[~df["session"].isin(GPS_ONLY_SESSIONS)].copy()

land_cover = load_land_cover()
df["tempC_anomaly"] = df["tempC"] - df["tempC"].mean()

# Urban Heat Island index (drift-corrected): hardscape mean minus green canopy mean
_hard_mean  = df_sensor[df_sensor["zone"].isin(["Museumplein", "Frans Halsbuurt"])]["tempC_detrended"].mean()
_green_mean = df_sensor[df_sensor["zone"] == "Sarphatipark"]["tempC_detrended"].mean()
uhi_index = (
    round(float(_hard_mean - _green_mean), 2)
    if (not np.isnan(_hard_mean) and not np.isnan(_green_mean))
    else None
)

session_order = [s for s in available_sessions if s in sessions]
# Sensor-only subset: GPS-track sessies hebben geen bruikbare sensordata
sensor_session_order = [s for s in session_order if s not in GPS_ONLY_SESSIONS]
is_compare = len(sensor_session_order) > 1

# --------------------------------------------------------------------------
# KNMI-weerdata ophalen voor elke geselecteerde sessie
# --------------------------------------------------------------------------
# Eerst de automatische API-fetch proberen, dan eventueel overschrijven met
# handmatige waarden. Fouten worden verzameld zodat we ze leesbaar kunnen
# tonen in plaats van de hele pagina laten crashen.
knmi_per_session: dict[str, dict] = {}
knmi_errors: dict[str, str] = {}

for s in sensor_session_order:
    sub = df_sensor[df_sensor["session"] == s]
    if sub.empty:
        continue

    # Handmatige overschrijving heeft prioriteit; alleen non-None velden worden gebruikt
    mk = manual_knmi.get(s, {})
    mk_filled = {k: v for k, v in mk.items() if v is not None}
    if use_manual and mk_filled:
        knmi_per_session[s] = {**mk_filled, "hours_covered": "handmatig ingevoerd"}
        continue

    # Anders automatische KNMI-fetch
    date_str = str(sub["timestamp"].iloc[0].date())
    try:
        with st.spinner(f"KNMI-weerdata ophalen voor {date_str}…"):
            knmi_df = fetch_knmi_weather(date_str)
        summary = knmi_summary_for_walk(sub, knmi_df)
        if summary:
            knmi_per_session[s] = summary
        else:
            knmi_errors[s] = ("KNMI-data ontvangen maar geen uren "
                              "overlappen met de wandeltijd.")
    except KNMIFetchError as e:
        knmi_errors[s] = str(e)

# Toon eventuele KNMI-fetch-fouten boven het dashboard zodat ze niet
# stilletjes onder de KPI-tabel verdwijnen
if knmi_errors:
    with st.expander(
        f"⚠️ KNMI-weerdata kon niet voor {len(knmi_errors)} sessie(s) "
        "opgehaald worden (klik voor details)",
        expanded=False,
    ):
        for s, err in knmi_errors.items():
            st.markdown(f"**{s}**: {err}")
        st.caption(
            "Voor zeer recente data (binnen ~1 week) heeft KNMI vaak "
            "validatie-vertraging. Tip: open de zijbalk → "
            "**🌤️ KNMI handmatig instellen** om waarden van bv. "
            "[weerlive.nl](https://weerlive.nl) of de weer-app van je "
            "telefoon in te voeren als referentie."
        )

# --------------------------------------------------------------------------
# Hero header
# --------------------------------------------------------------------------
if is_compare:
    _meta_line = (
        f"Vergelijking van <strong>{len(sessions)} sessies</strong> "
        f"({', '.join(sessions)}) &mdash; {len(df_sensor):,} sensor-metingen totaal."
    )
else:
    _only = sessions[0]
    _sub  = df_sensor[df_sensor["session"] == _only] if _only not in GPS_ONLY_SESSIONS else df[df["session"] == _only]
    _meta_line = (
        f"<strong>{_only}</strong> &mdash; {len(_sub):,} metingen van "
        f"{_sub['timestamp'].min():%Y-%m-%d %H:%M} tot {_sub['timestamp'].max():%H:%M}."
    )

st.markdown(f"""
<div class="hero-banner">
  <div style="display:flex;align-items:center;gap:1.5rem;">
    <span style="font-size:3.2rem;line-height:1;">📡</span>
    <div>
      <h1>Arduino Sensor Logger Dashboard</h1>
      <p class="hero-sub">
        Stadsmicroklimaat-veldwerk in Amsterdam &mdash;
        Museumplein, Frans Halsbuurt en Sarphatipark
      </p>
      <p class="hero-meta">{_meta_line}</p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

if df.empty:
    st.warning("Geen rijen voldoen aan de huidige filters.")
    st.stop()


# ==========================================================================
# HOOFDBEVINDINGEN — visueel verhaal bovenaan, altijd zichtbaar
# ==========================================================================
st.markdown("## 🔍 Hoofdbevindingen")

# --- Reken de cijfers voor de inzicht-kaarten ---
def _safe_mean(series): return series.dropna().mean() if len(series.dropna()) else np.nan

stats_per_session = {}
for s in sensor_session_order:
    sub = df_sensor[df_sensor["session"] == s].dropna(subset=["tempC"])
    if sub.empty:
        continue
    first10 = sub[sub["minute_from_start"] <= 10]["tempC"].mean()
    end_min = sub["minute_from_start"].max()
    last10  = sub[sub["minute_from_start"] >= end_min - 10]["tempC"].mean()
    stats_per_session[s] = {
        "n": len(sub),
        "mean": sub["tempC"].mean(),
        "drift": last10 - first10,
        "duration_min": end_min,
    }

# --- Kaart-layout: 4 inzicht-kaarten naast elkaar ---
col1, col2, col3, col4 = st.columns(4)

# Kaart 1: aantal metingen + dataset-overzicht
with col1:
    total_n = len(df_sensor)
    n_zones = df_sensor[df_sensor["zone"] != "Onderweg"]["zone"].nunique()
    st.markdown(
        f"""<div class="insight-card insight-card--blue">
          <p class="card-title">📊 Dataset</p>
          <p class="card-body">
            <strong>{total_n:,}</strong> metingen verzameld over
            <strong>{len(session_order)} sessie(s)</strong> in
            <strong>{n_zones} stadszones</strong>.
          </p>
        </div>""",
        unsafe_allow_html=True,
    )

# Kaart 2: temperatuur-drift (kalibratie-uitdaging)
with col2:
    if "Dag 1 - deksel dicht" in stats_per_session:
        d = stats_per_session["Dag 1 - deksel dicht"]
        drift_body = (
            f"Op Dag 1 steeg de gemeten temperatuur met "
            f"<strong>+{d['drift']:.1f} °C</strong> over {d['duration_min']:.0f} min &mdash; "
            f"vermoedelijk zelf-opwarming van het board, niet de stad zelf."
        )
    else:
        drift_body = "Selecteer beide sessies om de drift-vergelijking te zien."
    st.markdown(
        f"""<div class="insight-card insight-card--amber">
          <p class="card-title">🌡️ Sensor-drift</p>
          <p class="card-body">{drift_body}</p>
        </div>""",
        unsafe_allow_html=True,
    )

# Kaart 3: belangrijkste bevinding van de route-omkering (Museumplein test)
with col3:
    mp_per_session = {}
    for s in sensor_session_order:
        mp_sub = df_sensor[(df_sensor["session"] == s) & (df_sensor["zone"] == "Museumplein")]["tempC"].dropna()
        if len(mp_sub):
            mp_per_session[s] = mp_sub.mean()
    if len(mp_per_session) == 2:
        diff = list(mp_per_session.values())[1] - list(mp_per_session.values())[0]
        route_body = (
            f"Museumplein in beide volgordes: "
            f"<strong>{list(mp_per_session.values())[0]:.1f} °C</strong> vs "
            f"<strong>{list(mp_per_session.values())[1]:.1f} °C</strong> "
            f"(&Delta; = {diff:+.2f} °C). "
            f"Suggereert dat eerdere zone-effecten deels drift waren."
        )
    else:
        route_body = (
            "Dag 1 (Sarphatipark &rarr; Museumplein) vs Dag 2 (Museumplein &rarr; Sarphatipark): "
            "route-omkering controleert voor sensor-drift."
        )
    st.markdown(
        f"""<div class="insight-card insight-card--green">
          <p class="card-title">🔄 Route-omkering test</p>
          <p class="card-body">{route_body}</p>
        </div>""",
        unsafe_allow_html=True,
    )

# Kaart 4: bodemgebruik-verrassingsbevinding
with col4:
    _lc_rows_card = []
    for zone_name, lc_data in land_cover.items():
        kl = lc_data["klasse"]
        _lc_rows_card.append((zone_name, kl.get("verhard", 0), kl.get("groen", 0)))
    _most_verhard = max(_lc_rows_card, key=lambda x: x[1])
    _most_groen   = max(_lc_rows_card, key=lambda x: x[2])
    if _most_verhard[0] == _most_groen[0]:
        lc_body = (
            f"<strong>{_most_verhard[0]}</strong>: meeste verhard "
            f"(<strong>{_most_verhard[1]:.0f}%</strong>) én meeste groen "
            f"(<strong>{_most_verhard[2]:.0f}%</strong>). "
            f"Het UHI-effect zit in het asfalttype, niet in groengebrek."
        )
    else:
        lc_body = (
            f"Meest verhard: <strong>{_most_verhard[0]}</strong> ({_most_verhard[1]:.0f}%). "
            f"Meest groen: <strong>{_most_groen[0]}</strong> ({_most_groen[2]:.0f}%)."
        )
    st.markdown(
        f"""<div class="insight-card insight-card--blue">
          <p class="card-title">🗺️ Bodemgebruik</p>
          <p class="card-body">{lc_body}</p>
        </div>""",
        unsafe_allow_html=True,
    )

# --- Urban Heat Island index banner ---
if uhi_index is not None:
    _uhi_color   = "#ef4444" if uhi_index > 0 else "#10b981"
    _uhi_sign    = "warmer" if uhi_index > 0 else "koeler"
    _uhi_context = (
        f"Museumplein + Frans Halsbuurt is gemiddeld "
        f"<strong style='color:{_uhi_color}'>{abs(uhi_index):.2f} °C {_uhi_sign}</strong> "
        f"dan Sarphatipark — berekend op drift-gecorrigeerde temperatuur. "
        f"Op buurtniveau heeft Museumplein het hoogste aandeel verhard (42.6%) "
        f"maar ook het meeste groen (32.8%); Sarphatipark heeft op buurtniveau "
        f"minder groen (22.8%). Het verschil is daardoor eerder toe te schrijven "
        f"aan het hoge asfaltaandeel dan aan afwezigheid van groen."
    )
    st.markdown(f"""
    <div style="background:#1e293b; border:1px solid #334155; border-radius:14px;
                padding:1.1rem 1.6rem; margin:.8rem 0 1.2rem;
                display:flex; align-items:center; gap:1.4rem;">
      <span style="font-size:2.6rem; line-height:1;">🏙️</span>
      <div>
        <div style="color:#94a3b8; font-size:.85rem; font-weight:700;
                    text-transform:uppercase; letter-spacing:.07em; margin-bottom:.3rem;">
          Urban Heat Island Index &nbsp;·&nbsp; drift-gecorrigeerd
        </div>
        <div style="color:{_uhi_color}; font-size:2.1rem; font-weight:800; line-height:1.15;">
          {uhi_index:+.2f} °C
        </div>
        <div style="color:#94a3b8; font-size:.95rem; margin-top:.35rem; line-height:1.6;">
          {_uhi_context}
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# --- Kern-KPI-tabel: zichtbaar samengevat ---
st.markdown("### 📋 Samenvatting per sessie")

if is_compare:
    kpi_rows = {}
    for s in sensor_session_order:
        sub = df_sensor[df_sensor["session"] == s]
        sub_t = sub.dropna(subset=["tempC"])
        duration = (sub["timestamp"].max() - sub["timestamp"].min()).total_seconds() / 60
        gps_valid = sub.dropna(subset=["lat_dec"])
        distance_km = sub.groupby("session")["step_m"].apply(lambda x: x.sum() / 1000).iloc[0] \
                      if "step_m" in sub.columns else 0
        sensor_mean = sub_t["tempC"].mean()
        knmi = knmi_per_session.get(s, {})
        knmi_temp = knmi.get("temp_C")
        offset = (sensor_mean - knmi_temp) if knmi_temp is not None else None

        kpi_rows[s] = {
            "Aantal metingen":          f"{len(sub):,}",
            "Duur (min)":               f"{duration:.0f}",
            "Afstand (km)":             f"{distance_km:.2f}",
            "Gem. temperatuur sensor (°C)":  f"{sensor_mean:.1f}",
            "Std. temperatuur (°C)":    f"{sub_t['tempC'].std():.2f}",
            "Gem. luchtdruk (hPa)":     f"{sub['pressure_hPa'].mean():.1f}",
            "Max. licht (lux)":         f"{sub['light_lux'].max():,.0f}",
            "KNMI omgeving (°C)":       f"{knmi_temp:.1f}" if knmi_temp is not None else "—",
            "Sensor − KNMI offset (°C)": f"{offset:+.1f}" if offset is not None else "—",
            "KNMI wind (m/s)":          f"{knmi['wind_ms']:.1f}" if "wind_ms" in knmi else "—",
            "KNMI bewolking (okta 0–9)": f"{knmi['cloud_okta']:.1f}" if "cloud_okta" in knmi else "—",
        }
    st.dataframe(pd.DataFrame(kpi_rows), use_container_width=True)
else:
    sub = df_sensor if sensor_session_order else df
    sub_t = sub.dropna(subset=["tempC"])
    duration = (sub["timestamp"].max() - sub["timestamp"].min()).total_seconds() / 60 if not sub.empty else 0
    distance_km = sub["step_m"].sum() / 1000
    only_session = sensor_session_order[0] if sensor_session_order else sessions[0]
    knmi = knmi_per_session.get(only_session, {})

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Aantal metingen", f"{len(sub):,}")
    c2.metric("Duur", f"{duration:.0f} min")
    c3.metric("Afstand", f"{distance_km:.2f} km")
    sensor_mean = sub_t['tempC'].mean()
    if "temp_C" in knmi:
        offset = sensor_mean - knmi["temp_C"]
        c4.metric("Gem. temperatuur (sensor)",
                  f"{sensor_mean:.1f} °C",
                  delta=f"{offset:+.1f} vs KNMI ({knmi['temp_C']:.1f} °C)",
                  delta_color="off")
    else:
        c4.metric("Gem. temperatuur", f"{sensor_mean:.1f} °C")
    c5.metric("Piek licht", f"{sub['light_lux'].max():,.0f} lx")

# --- Headline-visualisatie: temperatuur over wandeling per sessie ---
st.markdown("### 📈 Temperatuur tijdens de wandeling")
st.caption(
    "De stippellijnen tonen de gemiddelde temperatuur per zone per sessie. "
    "Als zones consistent boven/onder de gemiddelde lijn liggen over verschillende "
    "tijdstippen heen, is dat het ruimtelijke signaal dat we zoeken."
)

headline_df = df_sensor.dropna(subset=["tempC"]).copy()
fig_headline = px.scatter(
    headline_df,
    x="minute_from_start", y="tempC",
    color="session" if is_compare else "zone",
    category_orders={
        "session": sensor_session_order,
        "zone":    [z for z in ZONE_COLOURS if z in headline_df["zone"].unique()],
    },
    color_discrete_map=SESSION_COLOURS if is_compare else ZONE_COLOURS,
    symbol="zone" if is_compare else None,
    opacity=0.65,
    template="plotly_dark",
    labels={
        "minute_from_start": "Minuten sinds start van de sessie",
        "tempC": "Temperatuur (°C)",
        "session": "Sessie",
        "zone": "Zone",
    },
)
fig_headline.update_traces(marker=dict(size=7))
fig_headline.update_layout(
    height=400,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#1e293b",
    margin=dict(l=10, r=10, t=10, b=10),
    legend_title_text="",
    hovermode="closest",
    xaxis=dict(gridcolor="#334155", zerolinecolor="#475569"),
    yaxis=dict(gridcolor="#334155", zerolinecolor="#475569"),
    legend=dict(bgcolor="rgba(30,41,59,0.85)", bordercolor="#334155", borderwidth=1),
)

# Voeg KNMI omgevingstemperatuur als horizontale referentielijn toe per sessie.
# Dit maakt de sensor-vs-omgeving offset visueel direct af te lezen.
for s in sensor_session_order:
    knmi = knmi_per_session.get(s, {})
    if "temp_C" in knmi:
        line_color = SESSION_COLOURS.get(s, "#888888") if is_compare else "#666666"
        fig_headline.add_hline(
            y=knmi["temp_C"],
            line_dash="dot",
            line_color=line_color,
            line_width=2,
            annotation_text=f"KNMI {s.split(' - ')[0]}: {knmi['temp_C']:.1f} °C",
            annotation_position="right",
            annotation_font=dict(size=10, color=line_color),
            opacity=0.6,
        )

st.plotly_chart(fig_headline, use_container_width=True)

if knmi_per_session:
    st.caption(
        "💡 De stippellijnen tonen de KNMI-omgevingstemperatuur (station Schiphol) "
        "tijdens elke wandeling. Het verschil tussen sensor-metingen en deze lijn "
        "is de **kalibratie-offset** — het deel van de gemeten temperatuur dat "
        "aan het meetinstrument zelf toebehoort, niet aan de stadsomgeving."
    )

# --------------------------------------------------------------------------
# Methodologische context per sessie
# --------------------------------------------------------------------------
with st.expander("📝 Methodologische context per sessie", expanded=False):
    st.caption(
        "Het experimentele ontwerp van elke wandeling wordt hier expliciet "
        "gedocumenteerd voor reproduceerbaarheid."
    )
    meta_cols = st.columns(max(len(sensor_session_order), 1))
    for col_st, s in zip(meta_cols, sensor_session_order):
        meta = SESSION_METADATA.get(s, {})
        knmi = knmi_per_session.get(s, {})
        with col_st:
            st.markdown(f"**{s}**")
            rows = [
                ("📅 Datum",            meta.get("datum", "—")),
                ("🚶 Wandelrichting",   meta.get("wandel_richting", "—")),
                ("📦 Deksel-config",    meta.get("deksel", "—")),
                ("🔌 Hardware-status",  meta.get("hardware_status", "—")),
                ("⚠️ Bekende issues",   meta.get("bekende_issues", "—")),
                ("🎯 Rol in analyse",   meta.get("rol_in_analyse", "—")),
            ]
            if knmi:
                rows.append(
                    ("🌤️ KNMI omstandigheden",
                     f"{knmi.get('temp_C', float('nan')):.1f}°C, "
                     f"wind {knmi.get('wind_ms', float('nan')):.1f} m/s, "
                     f"bewolking {knmi.get('cloud_okta', float('nan')):.0f}/9, "
                     f"straling {knmi.get('radiation_jcm2', float('nan')):.0f} J/cm²")
                )
            else:
                rows.append(("🌤️ KNMI omstandigheden",
                             "Niet beschikbaar (netwerk of geen data)"))
            for label, val in rows:
                st.markdown(f"- {label}: {val}")

st.divider()


# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
tab_time, tab_map, tab_zones, tab_corr, tab_data = st.tabs(
    ["📈 Tijdreeks", "🗺️ GPS-route", "🏛️ Zone-analyse", "🔗 Correlaties", "📋 Ruwe data"]
)

# ---- Tijdreeks ----------------------------------------------------------
with tab_time:
    st.subheader("Sensormetingen over tijd")
    if is_compare:
        st.caption(
            "X-as is **minuten sinds elke sessie start** zodat beide wandelingen "
            "op één tijdlijn overlappen, ondanks dat ze op verschillende dagen waren."
        )

    sensor_meta = {
        "tempC":            ("Temperatuur (°C)",                  "#ef4444"),
        "tempC_detrended":  ("Temp. drift-gecorrigeerd (°C)",     "#f97316"),
        "pressure_hPa":     ("Luchtdruk (hPa)",                   "#3b82f6"),
        "light_lux":        ("Licht (lux)",                       "#eab308"),
        "mq_raw":           ("MQ gassensor (ruw)",                "#10b981"),
    }

    selected = st.multiselect(
        "Welke sensoren tonen",
        options=list(sensor_meta.keys()),
        default=list(sensor_meta.keys()),
        format_func=lambda c: sensor_meta[c][0],
    )

    log_light = st.checkbox("Logaritmische schaal voor licht", value=True,
                            help="Verschil tussen binnen en buitenshuis is een factor 1000")

    for col in selected:
        label, sensor_color = sensor_meta[col]
        if is_compare:
            fig = px.line(
                df_sensor, x="minute_from_start", y=col,
                color="session",
                category_orders={"session": sensor_session_order},
                color_discrete_map=SESSION_COLOURS,
                template="plotly_dark",
                title=label,
                labels={"minute_from_start": "Minuten sinds start"},
            )
        else:
            fig = px.line(df_sensor, x="timestamp", y=col,
                          template="plotly_dark", title=label)
            fig.update_traces(line=dict(color=sensor_color, width=2))

        fig.update_layout(
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#1e293b",
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis=dict(title="Minuten sinds start" if is_compare else None,
                       gridcolor="#334155", zerolinecolor="#475569"),
            yaxis=dict(title=label, gridcolor="#334155", zerolinecolor="#475569"),
            legend=dict(title="", bgcolor="rgba(30,41,59,0.85)",
                        bordercolor="#334155", borderwidth=1),
        )
        if col == "light_lux" and log_light:
            fig.update_yaxes(type="log")
        st.plotly_chart(fig, use_container_width=True)


# ---- GPS-route -----------------------------------------------------------
with tab_map:
    st.subheader("GPS-route")

    # Map compare = alle geselecteerde sessies, ook GPS-only tracks
    is_map_compare = len(session_order) > 1

    gps = df.dropna(subset=["lat_dec", "lon_dec"]).reset_index(drop=True)

    if gps.empty:
        st.info("Geen GPS-fixes beschikbaar in dit venster.")
    else:
        # --- Route-statistieken ------------------------------------------
        if is_map_compare:
            metrics_rows = {}
            for s in session_order:
                ssub = gps[gps["session"] == s]
                if ssub.empty:
                    continue
                dur = (ssub["timestamp"].max() - ssub["timestamp"].min()).total_seconds() / 60
                metrics_rows[s] = {
                    "Afstand (km)":            f"{ssub['step_m'].sum() / 1000:.2f}",
                    "Gem. snelheid (km/h)":    f"{ssub['speed_kmh'].mean():.2f}",
                    "Piek snelheid p95":       f"{ssub['speed_kmh'].quantile(0.95):.2f}",
                    "Wandeltijd (min)":        f"{dur:.1f}",
                }
            st.dataframe(pd.DataFrame(metrics_rows), use_container_width=True)
        else:
            total_dist_m = gps["step_m"].sum(skipna=True)
            avg_speed   = gps["speed_kmh"].mean(skipna=True)
            peak_speed  = gps["speed_kmh"].quantile(0.95)
            track_min   = (gps["timestamp"].max() - gps["timestamp"].min()).total_seconds() / 60

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Afstand gelopen",  f"{total_dist_m/1000:.2f} km")
            m2.metric("Gem. snelheid",    f"{avg_speed:.2f} km/h")
            m3.metric("Piek snelheid (p95)", f"{peak_speed:.2f} km/h")
            m4.metric("Wandeltijd",       f"{track_min:.1f} min")

        # --- Weergave-opties ---------------------------------------------
        c1, c2, c3 = st.columns([2, 1, 1])
        if is_map_compare:
            colour_label = "Sessie"
            colour_col   = "session"
            c1.info("In vergelijkmodus is de kleur ingesteld op **sessie**.")
            show_heatmap = False
        else:
            colour_options = {
                "Tijdsverloop":                          "_time_idx",
                "Temperatuur-afwijking (Δ van gem.)":    "tempC_anomaly",
                "Temp. drift-gecorrigeerd (°C)":         "tempC_detrended",
                "Snelheid (km/h)":                       "speed_kmh",
                "Temperatuur (°C)":                      "tempC",
                "Luchtdruk (hPa)":                       "pressure_hPa",
                "Licht (lux)":                           "light_lux",
                "MQ gas (ruw)":                          "mq_raw",
                "Zone":                                  "zone",
            }
            colour_label = c1.selectbox("Route inkleuren op",
                                        list(colour_options.keys()), index=1)
            colour_col = colour_options[colour_label]
            show_heatmap = c3.checkbox(
                "🌡️ Heatmap",
                value=False,
                help=(
                    "Ruimtelijke interpolatie van drift-gecorrigeerde temperatuur. "
                    "Schat de temperatuurverdeling op onbemeten plekken op basis "
                    "van nabijgelegen metingen (KDE, gewogen naar °C)."
                ),
            )

        map_style_label = c2.selectbox("Kaartstijl",
                                       list(MAP_STYLES.keys()), index=0)
        map_style = MAP_STYLES[map_style_label]

        gps["_time_idx"] = np.arange(len(gps))

        # --- Bouw de kaart op (MapLibre-gebaseerde Scattermap) -----------
        fig = go.Figure()

        # Heatmap-laag (onderste laag, toegevoegd vóór route en zones)
        if show_heatmap:
            hm = gps.dropna(subset=["lat_dec", "lon_dec", "tempC_detrended"])
            if len(hm) >= 4:
                _zlo = float(hm["tempC_detrended"].quantile(0.05))
                _zhi = float(hm["tempC_detrended"].quantile(0.95))
                fig.add_trace(go.Densitymap(
                    lat=hm["lat_dec"],
                    lon=hm["lon_dec"],
                    z=hm["tempC_detrended"],
                    radius=30,
                    colorscale="Inferno",
                    zmin=_zlo,
                    zmax=_zhi,
                    showscale=True,
                    colorbar=dict(
                        title=dict(text="°C", side="right"),
                        thickness=14,
                        x=1.0,
                        len=0.6,
                    ),
                    opacity=0.6,
                    name="Temp. heatmap",
                    hoverinfo="skip",
                    showlegend=True,
                ))

        if is_map_compare:
            for s in session_order:
                ssub = gps[gps["session"] == s]
                if ssub.empty:
                    continue
                short_name = s.split(" - ")[0]
                _gps_only = s in GPS_ONLY_SESSIONS
                _colour = SESSION_COLOURS.get(s, "#888888")

                fig.add_trace(go.Scattermap(
                    lat=ssub["lat_dec"], lon=ssub["lon_dec"],
                    mode="lines",
                    line=dict(width=2 if _gps_only else 3, color=_colour),
                    opacity=0.55 if _gps_only else 1.0,
                    name=f"{s} — route",
                    hoverinfo="skip",
                    legendgroup=s,
                ))
                fig.add_trace(go.Scattermap(
                    lat=ssub["lat_dec"], lon=ssub["lon_dec"],
                    mode="markers",
                    marker=dict(size=5 if _gps_only else 7, color=_colour,
                                opacity=0.55 if _gps_only else 1.0),
                    name=f"{s} — GPS" if _gps_only else f"{s} — metingen",
                    hovertext=ssub["timestamp"].dt.strftime("%H:%M:%S")
                              if _gps_only else
                              ssub["timestamp"].dt.strftime("%H:%M:%S")
                              + " — " + ssub["tempC"].round(1).astype(str) + " °C",
                    hoverinfo="text",
                    legendgroup=s,
                ))

                start_pt = ssub.iloc[0]
                end_pt   = ssub.iloc[-1]
                fig.add_trace(go.Scattermap(
                    lat=[start_pt["lat_dec"]], lon=[start_pt["lon_dec"]],
                    mode="markers+text",
                    marker=dict(size=16, color="#10b981"),
                    text=[f"{short_name} start"], textposition="top right",
                    textfont=dict(size=12, color="#10b981"),
                    name=f"{short_name} start",
                    hovertext=f"{s} start: {start_pt['timestamp']:%H:%M:%S}",
                    hoverinfo="text",
                    legendgroup=s,
                ))
                fig.add_trace(go.Scattermap(
                    lat=[end_pt["lat_dec"]], lon=[end_pt["lon_dec"]],
                    mode="markers+text",
                    marker=dict(size=16, color="#ef4444"),
                    text=[f"{short_name} eind"], textposition="top right",
                    textfont=dict(size=12, color="#ef4444"),
                    name=f"{short_name} eind",
                    hovertext=f"{s} eind: {end_pt['timestamp']:%H:%M:%S}",
                    hoverinfo="text",
                    legendgroup=s,
                ))
        else:
            fig.add_trace(go.Scattermap(
                lat=gps["lat_dec"], lon=gps["lon_dec"],
                mode="lines",
                line=dict(width=4, color="#3b82f6"),
                name="Route",
                hoverinfo="skip",
            ))

            if colour_col == "zone":
                zone_order = [z_ for z_ in ZONE_COLOURS if z_ in gps["zone"].unique()]
                for zname in zone_order:
                    zsub = gps[gps["zone"] == zname]
                    fig.add_trace(go.Scattermap(
                        lat=zsub["lat_dec"], lon=zsub["lon_dec"],
                        mode="markers",
                        marker=dict(size=9, color=ZONE_COLOURS[zname]),
                        name=zname,
                        hovertext=zsub["timestamp"].dt.strftime("%H:%M:%S")
                                  + " — " + zsub["tempC"].round(1).astype(str) + " °C",
                        hoverinfo="text",
                    ))
            else:
                if colour_col == "tempC_anomaly":
                    cscale, cmid = "RdBu_r", 0
                else:
                    cscale, cmid = "Viridis", None

                fig.add_trace(go.Scattermap(
                    lat=gps["lat_dec"], lon=gps["lon_dec"],
                    mode="markers",
                    marker=dict(
                        size=9,
                        color=gps[colour_col],
                        colorscale=cscale,
                        cmid=cmid,
                        showscale=True,
                        colorbar=dict(title=colour_label),
                    ),
                    name="Metingen",
                    customdata=np.stack([
                        gps["timestamp"].dt.strftime("%H:%M:%S"),
                        gps["speed_kmh"].fillna(0),
                        gps["cum_dist_m"],
                        gps["tempC"].fillna(0),
                        gps["pressure_hPa"].fillna(0),
                        gps["light_lux"].fillna(0),
                        gps["mq_raw"].fillna(0),
                    ], axis=-1),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Snelheid: %{customdata[1]:.2f} km/h<br>"
                        "Afstand sinds start: %{customdata[2]:.0f} m<br>"
                        "Temp: %{customdata[3]:.1f} °C<br>"
                        "Druk: %{customdata[4]:.1f} hPa<br>"
                        "Licht: %{customdata[5]:.0f} lux<br>"
                        "MQ: %{customdata[6]:.0f}<extra></extra>"
                    ),
                ))

            start_pt = gps.iloc[0]
            end_pt   = gps.iloc[-1]
            fig.add_trace(go.Scattermap(
                lat=[start_pt["lat_dec"]], lon=[start_pt["lon_dec"]],
                mode="markers+text",
                marker=dict(size=18, color="#10b981"),
                text=["Start"], textposition="top right",
                textfont=dict(size=14, color="#10b981"),
                name="Start",
                hovertext=f"Start: {start_pt['timestamp']:%H:%M:%S}",
                hoverinfo="text",
            ))
            fig.add_trace(go.Scattermap(
                lat=[end_pt["lat_dec"]], lon=[end_pt["lon_dec"]],
                mode="markers+text",
                marker=dict(size=18, color="#ef4444"),
                text=["Eind"], textposition="top right",
                textfont=dict(size=14, color="#ef4444"),
                name="Eind",
                hovertext=f"Eind: {end_pt['timestamp']:%H:%M:%S}",
                hoverinfo="text",
            ))

        # --- Zone-vlakken (gevuld) en centroid-labels ---
        zones_wgs84 = zones_gdf.to_crs(WGS84)
        for _, zrow in zones_wgs84.iterrows():
            zname  = zrow["zone"]
            zcolor = ZONE_COLOURS.get(zname, "#ffffff")
            coords = list(zrow.geometry.exterior.coords)
            z_lons = [c[0] for c in coords] + [coords[0][0]]
            z_lats = [c[1] for c in coords] + [coords[0][1]]
            # Converteer hex → rgba voor halftransparante vulling
            _h = zcolor.lstrip("#")
            _r, _g, _b = int(_h[0:2], 16), int(_h[2:4], 16), int(_h[4:6], 16)
            fill_rgba = f"rgba({_r},{_g},{_b},0.18)"
            fig.add_trace(go.Scattermap(
                lat=z_lats, lon=z_lons,
                mode="lines",
                fill="toself",
                fillcolor=fill_rgba,
                line=dict(color=zcolor, width=2.5),
                name=f"Zone: {zname}",
                hoverinfo="skip",
                showlegend=False,
                opacity=0.9,
            ))
            centroid = zrow.geometry.centroid
            fig.add_trace(go.Scattermap(
                lat=[centroid.y], lon=[centroid.x],
                mode="text",
                text=[zname],
                textfont=dict(size=12, color=zcolor),
                hoverinfo="skip",
                showlegend=False,
            ))

        # Auto-zoom om de route te omsluiten
        center_lat = gps["lat_dec"].mean()
        center_lon = gps["lon_dec"].mean()
        span = max(
            gps["lat_dec"].max() - gps["lat_dec"].min(),
            gps["lon_dec"].max() - gps["lon_dec"].min(),
        )
        zoom = 15 if span < 0.01 else 14 if span < 0.03 else 13

        show_legend = is_map_compare or (not is_map_compare and colour_col == "zone")
        fig.update_layout(
            map=dict(style=map_style,
                     center=dict(lat=center_lat, lon=center_lon),
                     zoom=zoom),
            height=640,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=show_legend,
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(bgcolor="rgba(15,23,42,0.85)", bordercolor="#334155",
                        borderwidth=1, font=dict(color="#cbd5e1")),
        )
        # Activeer de ingebouwde modebar-acties expliciet zodat zoom-knoppen
        # en pan-controls altijd zichtbaar zijn.
        st.plotly_chart(
            fig, use_container_width=True,
            config={
                "scrollZoom": True,
                "displayModeBar": True,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            },
        )

        st.caption(
            "💡 Tip: scroll om in/uit te zoomen, sleep om te pannen. Klik op een sessie "
            "in de legenda om hem te verbergen."
        )

        # --- Snelheid-over-tijd grafiek ---------------------------------
        with st.expander("📊 Snelheid over tijd", expanded=False):
            if is_map_compare:
                speed_fig = px.line(
                    gps, x="minute_from_start", y="speed_kmh",
                    color="session",
                    category_orders={"session": session_order},
                    color_discrete_map=SESSION_COLOURS,
                    template="plotly_dark",
                    labels={"speed_kmh": "Snelheid (km/h)",
                            "minute_from_start": "Minuten sinds start"},
                )
            else:
                speed_fig = px.line(
                    gps, x="timestamp", y="speed_kmh",
                    template="plotly_dark",
                    labels={"speed_kmh": "Snelheid (km/h)", "timestamp": "Tijd"},
                )
                speed_fig.update_traces(line=dict(color="#3b82f6", width=2))
            speed_fig.update_layout(
                height=260,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#1e293b",
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(title="", bgcolor="rgba(30,41,59,0.85)",
                            bordercolor="#334155", borderwidth=1),
                xaxis=dict(gridcolor="#334155", zerolinecolor="#475569"),
                yaxis=dict(gridcolor="#334155", zerolinecolor="#475569"),
            )
            st.plotly_chart(speed_fig, use_container_width=True)
            st.caption(
                "Snelheden zijn afgeleid van haversine-afstand tussen opeenvolgende "
                "GPS-fixes (~5s uit elkaar). Wandeltempo is normaal 4–6 km/h; "
                "alles boven ~10 km/h is meestal GPS-ruis, geen echte beweging."
            )


# ---- Zone-analyse --------------------------------------------------------
with tab_zones:
    st.subheader("Microklimaat-vergelijking tussen zones")
    if is_compare:
        st.caption(
            "Elke GPS-fix is toegewezen aan een zone en een sessie. De hoofdvraag — "
            "*herhaalt het ruimtelijke patroon van Dag 1 zich op Dag 2?* — leeft in deze tab."
        )
    else:
        st.caption(
            "Elke GPS-fix wordt toegewezen aan een zone (Museumplein, Frans Halsbuurt, "
            "Sarphatipark) of *Onderweg* als hij buiten alle drie valt."
        )

    c1, c2 = st.columns(2)
    exclude_transit = c1.toggle("Sluit metingen 'Onderweg' uit", value=True,
                                help="Verberg GPS-fixes die niet in een van de 3 zones vallen")
    only_stationary = c2.toggle("Alleen stilstaande metingen (< 0.5 km/h)", value=False,
                                help="Sensor heeft ~15s nodig om te stabiliseren tijdens lopen — "
                                     "stilstaande metingen zijn betrouwbaarder")

    z = df_sensor.copy()
    if exclude_transit:
        z = z[z["zone"] != "Onderweg"]
    if only_stationary:
        z = z[z["speed_kmh"].fillna(0) < 0.5]

    if len(z) == 0:
        st.warning("Geen metingen voldoen aan de huidige filters.")
    else:
        zone_order = [z_ for z_ in ZONE_COLOURS if z_ in z["zone"].unique()]

        # --- Overzicht-tabel ---------------------------------------------
        group_cols = ["zone", "session"] if is_compare else ["zone"]
        summary = (z.groupby(group_cols)
                    .agg(metingen=("tempC", "count"),
                         gem_temp=("tempC", "mean"),
                         std_temp=("tempC", "std"),
                         min_temp=("tempC", "min"),
                         max_temp=("tempC", "max"),
                         mediaan_licht=("light_lux", "median"),
                         max_licht=("light_lux", "max"),
                         gem_mq=("mq_raw", "mean"))
                    .round(2))
        st.markdown("**Samenvatting per zone**" + (" × sessie" if is_compare else ""))
        st.dataframe(summary, use_container_width=True)

        # --- Boxplot van temperatuur -------------------------------------
        st.markdown("**Temperatuurverdeling per zone**")
        _box_base = dict(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#1e293b",
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(gridcolor="#334155", zerolinecolor="#475569"),
            yaxis=dict(gridcolor="#334155", zerolinecolor="#475569"),
            legend=dict(title="", bgcolor="rgba(30,41,59,0.85)",
                        bordercolor="#334155", borderwidth=1),
        )
        if is_compare:
            fig_box = px.box(
                z, x="zone", y="tempC", color="session",
                category_orders={"zone": zone_order, "session": sensor_session_order},
                color_discrete_map=SESSION_COLOURS,
                points="all",
                template="plotly_dark",
                labels={"tempC": "Temperatuur (°C)", "zone": "", "session": ""},
            )
            fig_box.update_layout(**_box_base, boxmode="group", height=420)
        else:
            fig_box = px.box(
                z, x="zone", y="tempC", color="zone",
                category_orders={"zone": zone_order},
                color_discrete_map=ZONE_COLOURS,
                points="all",
                template="plotly_dark",
                labels={"tempC": "Temperatuur (°C)", "zone": ""},
            )
            fig_box.update_layout(**_box_base, showlegend=False, height=400)
        st.plotly_chart(fig_box, use_container_width=True)

        if is_compare:
            st.caption(
                "Als dezelfde zone-volgorde geldt over beide sessies (bv. Frans Halsbuurt "
                "het warmst in beide, Sarphatipark het koelst in beide), dan zijn de ruimtelijke "
                "bevinding robuust voor de wijziging in methodologie."
            )

        # --- Gemiddelde temperatuur per zone ---------------------------------
        st.markdown("### 🌡️ Gemiddelde temperatuur per zone")
        st.caption(
            "Gemiddelde drift-gecorrigeerde temperatuur per zone. "
            "Drift-correctie verwijdert de geleidelijke opwarming van de sensor "
            "zelf, zodat het ruimtelijke signaal overblijft."
        )
        _mean_rows = []
        for zone_name in zone_order:
            sub_z = z[z["zone"] == zone_name]
            if is_compare:
                for s in sensor_session_order:
                    sub_zs = sub_z[sub_z["session"] == s]["tempC_detrended"].dropna()
                    if len(sub_zs):
                        _mean_rows.append({
                            "zone": zone_name, "sessie": s,
                            "gem_temp": round(sub_zs.mean(), 2),
                            "n": len(sub_zs),
                        })
            else:
                sub_zs = sub_z["tempC_detrended"].dropna()
                if len(sub_zs):
                    _mean_rows.append({
                        "zone": zone_name,
                        "gem_temp": round(sub_zs.mean(), 2),
                        "n": len(sub_zs),
                    })

        if _mean_rows:
            _mean_df = pd.DataFrame(_mean_rows)
            if is_compare:
                fig_mean = px.bar(
                    _mean_df, x="zone", y="gem_temp", color="sessie",
                    barmode="group",
                    category_orders={"zone": zone_order, "sessie": sensor_session_order},
                    color_discrete_map=SESSION_COLOURS,
                    template="plotly_dark",
                    labels={"gem_temp": "Gem. temperatuur (°C)", "zone": "", "sessie": ""},
                    text="gem_temp",
                )
            else:
                fig_mean = px.bar(
                    _mean_df, x="zone", y="gem_temp", color="zone",
                    category_orders={"zone": zone_order},
                    color_discrete_map=ZONE_COLOURS,
                    template="plotly_dark",
                    labels={"gem_temp": "Gem. temperatuur (°C)", "zone": ""},
                    text="gem_temp",
                )
                fig_mean.update_layout(showlegend=False)
            fig_mean.update_traces(texttemplate="%{text:.2f} °C", textposition="outside",
                                   textfont=dict(size=11))
            fig_mean.update_layout(
                height=360,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#1e293b",
                margin=dict(l=10, r=10, t=30, b=10),
                xaxis=dict(gridcolor="#334155"),
                yaxis=dict(gridcolor="#334155"),
                legend=dict(title="", bgcolor="rgba(30,41,59,0.85)",
                            bordercolor="#334155", borderwidth=1),
            )
            st.plotly_chart(fig_mean, use_container_width=True)

        # --- Licht vs Temperatuur scatter --------------------------------
        st.markdown("**Licht vs temperatuur**")
        if is_compare:
            fig_sc = px.scatter(
                z.dropna(subset=["light_lux", "tempC"]),
                x="light_lux", y="tempC",
                color="session",
                facet_col="zone",
                category_orders={"zone": zone_order, "session": sensor_session_order},
                color_discrete_map=SESSION_COLOURS,
                log_x=True, opacity=0.7,
                template="plotly_dark",
                labels={"light_lux": "Licht (lux, log)", "tempC": "Temperatuur (°C)"},
            )
        else:
            fig_sc = px.scatter(
                z.dropna(subset=["light_lux", "tempC"]),
                x="light_lux", y="tempC", color="zone",
                category_orders={"zone": zone_order},
                color_discrete_map=ZONE_COLOURS,
                log_x=True, opacity=0.7,
                template="plotly_dark",
                labels={"light_lux": "Licht (lux, log)", "tempC": "Temperatuur (°C)"},
            )
        fig_sc.update_traces(marker=dict(size=8))
        fig_sc.update_layout(
            height=420,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#1e293b",
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(title="", bgcolor="rgba(30,41,59,0.85)",
                        bordercolor="#334155", borderwidth=1),
        )
        st.plotly_chart(fig_sc, use_container_width=True)

        # --- Tijd-van-dag confounder check -------------------------------
        st.markdown("**Wanneer is elke zone bezocht?** *(tijd-van-dag controle)*")
        _tl_base = dict(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#1e293b",
            showlegend=False,
            xaxis=dict(gridcolor="#334155", zerolinecolor="#475569"),
            yaxis=dict(gridcolor="#334155", zerolinecolor="#475569"),
        )
        if is_compare:
            timeline = (z.assign(minute=z["minute_from_start"].round())
                         .groupby(["minute", "zone", "session"])
                         .size().reset_index(name="metingen"))
            fig_tl = px.scatter(
                timeline, x="minute", y="zone",
                size="metingen", color="zone",
                facet_row="session",
                category_orders={"zone": zone_order, "session": sensor_session_order},
                color_discrete_map=ZONE_COLOURS,
                template="plotly_dark",
                labels={"minute": "Minuten sinds start", "zone": ""},
            )
            fig_tl.update_layout(**_tl_base, height=320, margin=dict(l=10, r=10, t=30, b=10))
        else:
            timeline = (z.assign(minute=z["timestamp"].dt.floor("1min"))
                         .groupby(["minute", "zone"]).size().reset_index(name="metingen"))
            fig_tl = px.scatter(
                timeline, x="minute", y="zone", size="metingen", color="zone",
                category_orders={"zone": zone_order},
                color_discrete_map=ZONE_COLOURS,
                template="plotly_dark",
                labels={"minute": "Tijd", "zone": ""},
            )
            fig_tl.update_layout(**_tl_base, height=220, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_tl, use_container_width=True)
        st.caption(
            "Als zones in volgorde bezocht zijn (niet door elkaar), kan een deel van het "
            "temperatuurverschil tijd-van-dag zijn in plaats van locatie. Vermeld dit "
            "voorbehoud in je discussie."
        )

        # --- Bodemgebruik per zone ----------------------------------------
        st.markdown("### 🌿 Bodemgebruik per zone")
        st.caption(
            "Oppervlakteaandelen op basis van het BGT-bodemgebruiksbestand. "
            "Meer verhard oppervlak betekent meer warmte-absorptie overdag en "
            "minder verdampingskoeling — dit is de fysische basis van het "
            "Urban Heat Island effect dat we meten."
        )

        _klasse_colours = {
            "verhard":  "#94a3b8",
            "gebouw":   "#64748b",
            "groen":    "#22c55e",
            "water":    "#38bdf8",
        }
        lc_rows = []
        for zone_name, lc_data in land_cover.items():
            for klasse, pct in lc_data["klasse"].items():
                lc_rows.append({"zone": zone_name, "klasse": klasse, "percentage": pct})
        lc_df = pd.DataFrame(lc_rows)

        fig_lc = px.bar(
            lc_df,
            x="zone", y="percentage", color="klasse",
            category_orders={
                "zone":   list(ZONE_COLOURS.keys()),
                "klasse": ["verhard", "gebouw", "groen", "water"],
            },
            color_discrete_map=_klasse_colours,
            template="plotly_dark",
            labels={"percentage": "Oppervlakteaandeel (%)", "zone": "", "klasse": ""},
            text="percentage",
        )
        fig_lc.update_traces(texttemplate="%{text:.0f}%", textposition="inside",
                             textfont=dict(size=11))
        fig_lc.update_layout(
            barmode="stack",
            height=340,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#1e293b",
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(title="", bgcolor="rgba(30,41,59,0.85)",
                        bordercolor="#334155", borderwidth=1),
            xaxis=dict(gridcolor="#334155"),
            yaxis=dict(gridcolor="#334155", range=[0, 102]),
        )
        st.plotly_chart(fig_lc, use_container_width=True)

        # Koppel bodemgebruik aan gemeten temperatuur
        _lc_temp_rows = []
        for zone_name in land_cover:
            zone_sub = z[z["zone"] == zone_name]["tempC_detrended"].dropna()
            lc = land_cover[zone_name]["klasse"]
            if len(zone_sub):
                _lc_temp_rows.append({
                    "Zone":              zone_name,
                    "% verhard":         lc.get("verhard", 0),
                    "% groen":           lc.get("groen",   0),
                    "Gem. temp. (°C)":   round(zone_sub.mean(), 2),
                    "n metingen":        len(zone_sub),
                })
        if _lc_temp_rows:
            st.dataframe(
                pd.DataFrame(_lc_temp_rows).set_index("Zone"),
                use_container_width=True,
            )

        # --- Verharding detail: gesloten vs open --------------------------
        st.markdown("### 🏗️ Verharding: gesloten vs. open")
        st.caption(
            "Gesloten verharding (asfalt, tegels) absorbeert meer warmte en laat "
            "geen water door — een directe drijfveer van het Urban Heat Island effect. "
            "Open verharding (klinkers, grind) laat water door en koelt sneller af. "
            "Percentages zijn berekend binnen het verharde oppervlak per zone."
        )
        _v_colours = {
            "gesloten verharding": "#475569",
            "open verharding":     "#94a3b8",
            "overig":              "#cbd5e1",
        }
        _v_rows = []
        for zone_name, lc_data in land_cover.items():
            for sub, pct in lc_data["verharding_sub"].items():
                _v_rows.append({"zone": zone_name, "type": sub, "percentage": pct})
        fig_v = px.bar(
            pd.DataFrame(_v_rows),
            x="zone", y="percentage", color="type",
            category_orders={
                "zone": list(ZONE_COLOURS.keys()),
                "type": ["gesloten verharding", "open verharding", "overig"],
            },
            color_discrete_map=_v_colours,
            template="plotly_dark",
            labels={"percentage": "% binnen verhard oppervlak", "zone": "", "type": ""},
            text="percentage",
        )
        fig_v.update_traces(texttemplate="%{text:.0f}%", textposition="inside",
                            textfont=dict(size=11))
        fig_v.update_layout(
            barmode="stack", height=300,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#1e293b",
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(title="", bgcolor="rgba(30,41,59,0.85)",
                        bordercolor="#334155", borderwidth=1),
            xaxis=dict(gridcolor="#334155"),
            yaxis=dict(gridcolor="#334155", range=[0, 102]),
        )
        st.plotly_chart(fig_v, use_container_width=True)

        # --- Groen detail: groenvoorziening vs erf ------------------------
        st.markdown("### 🌳 Groen: openbaar park vs. privétuin")
        st.caption(
            "Openbaar groen (groenvoorziening) omvat parken, plantsoen en bermen — "
            "de plekken met bomen en grote aaneengesloten vegetatie die het sterkst "
            "koelen via verdamping en schaduw. Privétuinen (erf) zijn versnipperd "
            "en koelen minder effectief. "
            "Percentages zijn berekend binnen het groene oppervlak per zone."
        )
        _g_colours = {
            "groenvoorziening": "#16a34a",
            "erf":              "#86efac",
            "overig":           "#bbf7d0",
        }
        _g_rows = []
        for zone_name, lc_data in land_cover.items():
            for sub, pct in lc_data["groen_sub"].items():
                _g_rows.append({"zone": zone_name, "type": sub, "percentage": pct})
        fig_g = px.bar(
            pd.DataFrame(_g_rows),
            x="zone", y="percentage", color="type",
            category_orders={
                "zone": list(ZONE_COLOURS.keys()),
                "type": ["groenvoorziening", "erf", "overig"],
            },
            color_discrete_map=_g_colours,
            template="plotly_dark",
            labels={"percentage": "% binnen groen oppervlak", "zone": "", "type": ""},
            text="percentage",
        )
        fig_g.update_traces(texttemplate="%{text:.0f}%", textposition="inside",
                            textfont=dict(size=11))
        fig_g.update_layout(
            barmode="stack", height=300,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#1e293b",
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(title="", bgcolor="rgba(30,41,59,0.85)",
                        bordercolor="#334155", borderwidth=1),
            xaxis=dict(gridcolor="#334155"),
            yaxis=dict(gridcolor="#334155", range=[0, 102]),
        )
        st.plotly_chart(fig_g, use_container_width=True)


# ---- Correlaties ---------------------------------------------------------
with tab_corr:
    st.subheader("Hoe verhouden sensoren zich tot elkaar?")

    sensor_cols = ["tempC", "pressure_hPa", "light_lux", "mq_raw"]
    sensor_labels_nl = {
        "tempC": "Temperatuur",
        "pressure_hPa": "Luchtdruk",
        "light_lux": "Licht",
        "mq_raw": "MQ gas",
    }

    _hm_base = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#1e293b",
        font=dict(color="#cbd5e1"),
        coloraxis_colorbar=dict(tickfont=dict(color="#94a3b8")),
    )
    if is_compare:
        st.markdown("**Correlatiematrix per sessie**")
        cols_st = st.columns(len(sensor_session_order))
        for col_st, s in zip(cols_st, sensor_session_order):
            sub = df_sensor[df_sensor["session"] == s]
            corr = sub[sensor_cols].corr()
            corr.index = corr.index.map(sensor_labels_nl)
            corr.columns = corr.columns.map(sensor_labels_nl)
            fig = px.imshow(
                corr, text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1, aspect="auto",
                template="plotly_dark",
                title=s,
            )
            fig.update_layout(**_hm_base, height=380, margin=dict(l=10, r=10, t=40, b=10))
            col_st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Als een correlatie sterk verschilt tussen sessies, is dat een aanwijzing dat "
            "de deksel-wijziging de sensor-kruisgevoeligheid heeft beïnvloed (bv. "
            "licht→temp koppeling)."
        )
    else:
        corr = df_sensor[sensor_cols].corr()
        corr.index = corr.index.map(sensor_labels_nl)
        corr.columns = corr.columns.map(sensor_labels_nl)
        fig = px.imshow(
            corr, text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1, aspect="auto",
            template="plotly_dark",
        )
        fig.update_layout(**_hm_base, height=400, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Paar-scatter**")
    c1, c2 = st.columns(2)
    x_axis = c1.selectbox("X-as", sensor_cols, index=0,
                          format_func=lambda c: sensor_labels_nl[c])
    y_axis = c2.selectbox("Y-as", sensor_cols, index=2,
                          format_func=lambda c: sensor_labels_nl[c])

    if is_compare:
        fig2 = px.scatter(
            df_sensor, x=x_axis, y=y_axis,
            color="session",
            category_orders={"session": sensor_session_order},
            color_discrete_map=SESSION_COLOURS,
            trendline="ols",
            opacity=0.6, height=420,
            template="plotly_dark",
            labels={x_axis: sensor_labels_nl[x_axis], y_axis: sensor_labels_nl[y_axis]},
        )
    else:
        fig2 = px.scatter(
            df_sensor, x=x_axis, y=y_axis,
            trendline="ols",
            opacity=0.6, height=420,
            template="plotly_dark",
            labels={x_axis: sensor_labels_nl[x_axis], y_axis: sensor_labels_nl[y_axis]},
        )
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#1e293b",
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(title="", bgcolor="rgba(30,41,59,0.85)",
                    bordercolor="#334155", borderwidth=1),
        xaxis=dict(gridcolor="#334155", zerolinecolor="#475569"),
        yaxis=dict(gridcolor="#334155", zerolinecolor="#475569"),
    )
    st.plotly_chart(fig2, use_container_width=True)



with tab_data:
    st.subheader("Gefilterde records")
    st.dataframe(df, use_container_width=True, height=500)

    csv_bytes = df.to_csv(index=False).encode()
    fname = "sessies_gecombineerd.csv" if is_compare else f"{sessions[0].replace(' ', '_')}.csv"
    st.download_button(
        "⬇️ Download gefilterde CSV",
        data=csv_bytes,
        file_name=fname,
        mime="text/csv",
    )