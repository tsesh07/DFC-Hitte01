"""
Arduino Sensor Logger Dashboard
================================
Leest de schoongemaakte multi-sessie DataFrame via data_loader.load_data() en
visualiseert de GPS-route + omgevingssensor-metingen, met ondersteuning voor
het vergelijken van Dag 1 (deksel dicht) vs Dag 2 (deksel open).

Starten met:  streamlit run app.py
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Statistische toetsen
from scipy import stats as scipy_stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Voor KNMI-API requests
import urllib.request
import urllib.parse
import urllib.error
import io

from data_loader import load_data, SESSION_METADATA

# --------------------------------------------------------------------------
# Pagina-configuratie
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Sensor Logger Dashboard",
    page_icon="📡",
    layout="wide",
)

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

# --------------------------------------------------------------------------
# Zone-definities (Amsterdam) — pas centra/stralen aan voor je transect
# --------------------------------------------------------------------------
ZONES = {
    "Museumplein":     {"lon": 4.8810, "lat": 52.3580, "radius_m": 200, "surface": "verhard"},
    "Frans Halsbuurt": {"lon": 4.8920, "lat": 52.3563, "radius_m": 150, "surface": "verhard"},
    "Sarphatipark":    {"lon": 4.8950, "lat": 52.3540, "radius_m": 120, "surface": "boomkroon"},
}
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
@st.cache_data
def build_zones_gdf() -> gpd.GeoDataFrame:
    """Bouw een GeoDataFrame van zone-polygonen.

    Pipeline:
      1. Maak centrum-Points in WGS84 (lat/lon, graden).
      2. Herproject naar RD New (meters) zodat .buffer() in meters werkt.
      3. Vervang elk centrum door een circulaire buffer met de gewenste straal.
    """
    rows = [{"zone": name, **spec} for name, spec in ZONES.items()]
    gdf = gpd.GeoDataFrame(
        rows,
        geometry=[Point(r["lon"], r["lat"]) for r in rows],
        crs=WGS84,
    ).to_crs(RD_NEW)

    gdf["geometry"] = gdf.apply(lambda row: row.geometry.buffer(row["radius_m"]), axis=1)
    return gdf[["zone", "surface", "geometry"]]


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
# Statistische toetsen
# --------------------------------------------------------------------------
def run_zone_anova(subset_df: pd.DataFrame,
                   value_col: str = "tempC",
                   group_col: str = "zone") -> dict | None:
    """One-way ANOVA voor `value_col ~ group_col`, met η² effectgrootte.

    Returnt dict met F, p, eta_squared, n_groups, n_samples, of None
    als er te weinig groepen/data zijn voor een geldige toets.

    η² (eta-squared) interpretatie (Cohen 1988):
        0.01 = klein, 0.06 = middel, 0.14 = groot effect.
    """
    sub = subset_df.dropna(subset=[value_col, group_col])
    groups_data = []
    group_names = []
    for name, g in sub.groupby(group_col):
        if len(g) >= 2:
            groups_data.append(g[value_col].values)
            group_names.append(name)
    if len(groups_data) < 2:
        return None

    f_stat, p_value = scipy_stats.f_oneway(*groups_data)

    all_values = np.concatenate(groups_data)
    grand_mean = all_values.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups_data)
    ss_total = float(((all_values - grand_mean) ** 2).sum())
    eta_sq = ss_between / ss_total if ss_total > 0 else 0.0

    return {
        "F": float(f_stat),
        "p": float(p_value),
        "eta_squared": float(eta_sq),
        "n_groups": len(groups_data),
        "n_samples": int(len(all_values)),
        "group_names": group_names,
    }


def run_tukey_hsd(subset_df: pd.DataFrame,
                  value_col: str = "tempC",
                  group_col: str = "zone") -> pd.DataFrame | None:
    """Tukey HSD post-hoc voor paarsgewijze zone-vergelijkingen."""
    sub = subset_df.dropna(subset=[value_col, group_col])
    if sub[group_col].nunique() < 2:
        return None
    tukey = pairwise_tukeyhsd(sub[value_col], sub[group_col])
    return pd.DataFrame(tukey._results_table.data[1:],
                        columns=tukey._results_table.data[0])


def interpret_eta_squared(eta_sq: float) -> str:
    """Vertaal η² naar Cohen's standaardlabels."""
    if eta_sq < 0.01:
        return "verwaarloosbaar"
    if eta_sq < 0.06:
        return "klein"
    if eta_sq < 0.14:
        return "middel"
    return "groot"


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
    "**Dag 2**: intermittente USB-stroomstoringen — voorzichtig interpreteren."
)

# --------------------------------------------------------------------------
# Handmatige KNMI-overschrijving (fallback wanneer API niet werkt)
# --------------------------------------------------------------------------
# KNMI valideert zeer recente uurgegevens vaak pas na enkele dagen, dus voor
# wandelingen van afgelopen week werkt de automatische fetch niet altijd.
# Hier kan je de KNMI-waarden handmatig invoeren — bv. afgelezen van
# weerlive.nl, de KNMI-website, of een weerstation in de buurt.
with st.sidebar.expander("🌤️ KNMI handmatig instellen", expanded=False):
    st.caption(
        "Wanneer de automatische KNMI-fetch leeg blijft (recente data is "
        "vaak nog niet gevalideerd), kun je hier zelf de waarden invullen "
        "voor de wandel-uren. Deze overschrijven de automatische data."
    )
    manual_knmi: dict[str, dict] = {}
    use_manual = st.toggle("Handmatige waarden gebruiken", value=False)
    if use_manual:
        for s in available_sessions:
            st.markdown(f"**{s}**")
            t = st.number_input(f"Temperatuur (°C) — {s}",
                                value=16.0, step=0.1, key=f"knmi_t_{s}")
            w = st.number_input(f"Wind (m/s) — {s}",
                                value=3.0, step=0.1, key=f"knmi_w_{s}")
            c = st.slider(f"Bewolking (okta 0–9) — {s}",
                          0, 9, 4, key=f"knmi_c_{s}")
            manual_knmi[s] = {
                "temp_C":         t,
                "wind_ms":        w,
                "cloud_okta":     float(c),
                "radiation_jcm2": float("nan"),  # niet aflesbaar van weerapps
                "hours_covered":  "handmatig ingevoerd",
            }

# --------------------------------------------------------------------------
# Handmatige KNMI-invoer (fallback wanneer auto-fetch faalt)
# --------------------------------------------------------------------------
# KNMI's klimatologische uurgegevens hebben een vertraging van enkele dagen
# (validatie tijdens kantooruren). Voor recente metingen kun je hier de
# omgevingsgegevens handmatig invoeren — bv. afgelezen van Weeronline of
# je iPhone-weerapp tijdens de wandeling. Deze waarden overschrijven de
# auto-fetch als ze ingevuld zijn.
with st.sidebar.expander("🌤️ KNMI handmatig invoeren (fallback)", expanded=False):
    st.caption(
        "Vul in als de automatische KNMI-fetch faalt. Lege velden gebruiken "
        "de auto-fetch (indien beschikbaar)."
    )
    manual_knmi: dict[str, dict] = {}
    for s in available_sessions:
        st.markdown(f"**{s}**")
        c_a, c_b = st.columns(2)
        manual_knmi[s] = {
            "temp_C":     c_a.number_input(
                "Temp °C", value=None, placeholder="auto",
                key=f"mk_temp_{s}", step=0.1, format="%.1f"),
            "wind_ms":    c_b.number_input(
                "Wind m/s", value=None, placeholder="auto",
                key=f"mk_wind_{s}", step=0.1, format="%.1f"),
            "cloud_okta": c_a.number_input(
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

zones_gdf = build_zones_gdf()
df["zone"] = assign_zones_via_sjoin(df, zones_gdf)
df["tempC_anomaly"] = df["tempC"] - df["tempC"].mean()

is_compare = len(sessions) > 1
session_order = [s for s in available_sessions if s in sessions]

# --------------------------------------------------------------------------
# KNMI-weerdata ophalen voor elke geselecteerde sessie
# --------------------------------------------------------------------------
# Eerst de automatische API-fetch proberen, dan eventueel overschrijven met
# handmatige waarden. Fouten worden verzameld zodat we ze leesbaar kunnen
# tonen in plaats van de hele pagina laten crashen.
knmi_per_session: dict[str, dict] = {}
knmi_errors: dict[str, str] = {}

for s in session_order:
    sub = df[df["session"] == s]
    if sub.empty:
        continue

    # Eerst handmatige overschrijving controleren — heeft prioriteit
    if use_manual and s in manual_knmi:
        knmi_per_session[s] = manual_knmi[s]
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
# Header
# --------------------------------------------------------------------------
st.title("📡 Arduino Sensor Logger Dashboard")
st.markdown(
    "Stadsmicroklimaat-veldwerk in Amsterdam — Museumplein, Frans Halsbuurt en Sarphatipark."
)

if is_compare:
    st.caption(
        f"Vergelijking van **{len(sessions)} sessies** "
        f"({', '.join(sessions)}) — {len(df):,} metingen totaal."
    )
else:
    only = sessions[0]
    sub = df[df["session"] == only]
    st.caption(
        f"**{only}** — {len(sub):,} metingen van "
        f"{sub['timestamp'].min():%Y-%m-%d %H:%M:%S} tot {sub['timestamp'].max():%H:%M:%S}."
    )

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
for s in session_order:
    sub = df[df["session"] == s].dropna(subset=["tempC"])
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

# --- Kaart-layout: 3 inzicht-kaarten naast elkaar ---
col1, col2, col3 = st.columns(3)

# Kaart 1: aantal metingen + dataset-overzicht
with col1:
    total_n = len(df)
    n_zones = df[df["zone"] != "Onderweg"]["zone"].nunique()
    st.info(
        f"### 📊 Dataset\n"
        f"**{total_n:,}** metingen verzameld over **{len(session_order)} sessie(s)** "
        f"in **{n_zones} stadszones**."
    )

# Kaart 2: temperatuur-drift (kalibratie-uitdaging)
with col2:
    if "Dag 1 - deksel dicht" in stats_per_session:
        d = stats_per_session["Dag 1 - deksel dicht"]
        drift_text = f"**+{d['drift']:.1f} °C** over {d['duration_min']:.0f} min"
        st.warning(
            f"### 🌡️ Sensor-drift\n"
            f"Op Dag 1 steeg de gemeten temperatuur met {drift_text} — "
            f"vermoedelijk zelf-opwarming van het board, niet de stad zelf."
        )
    else:
        st.warning(
            f"### 🌡️ Sensor-kalibratie\n"
            f"Selecteer beide sessies om de drift-vergelijking te zien."
        )

# Kaart 3: belangrijkste bevinding van de route-omkering (Museumplein test)
with col3:
    mp_per_session = {}
    for s in session_order:
        mp_sub = df[(df["session"] == s) & (df["zone"] == "Museumplein")]["tempC"].dropna()
        if len(mp_sub):
            mp_per_session[s] = mp_sub.mean()
    if len(mp_per_session) == 2:
        diff = list(mp_per_session.values())[1] - list(mp_per_session.values())[0]
        st.success(
            f"### 🔄 Route-omkering test\n"
            f"Museumplein in beide volgordes: **{list(mp_per_session.values())[0]:.1f} °C** "
            f"vs **{list(mp_per_session.values())[1]:.1f} °C** "
            f"(Δ = {diff:+.2f} °C). "
            f"Suggereert dat eerdere zone-effecten deels drift waren."
        )
    else:
        st.success(
            f"### 🔄 Methodologie\n"
            f"Dag 1 (Sarphatipark → Museumplein) vs Dag 2 (Museumplein → Sarphatipark): "
            f"route-omkering controleert voor sensor-drift."
        )

# --- Kern-KPI-tabel: zichtbaar samengevat ---
st.markdown("### 📋 Samenvatting per sessie")

if is_compare:
    kpi_rows = {}
    for s in session_order:
        sub = df[df["session"] == s]
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
    sub = df
    sub_t = sub.dropna(subset=["tempC"])
    duration = (sub["timestamp"].max() - sub["timestamp"].min()).total_seconds() / 60
    distance_km = sub["step_m"].sum() / 1000
    only_session = sessions[0]
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

headline_df = df.dropna(subset=["tempC"]).copy()
fig_headline = px.scatter(
    headline_df,
    x="minute_from_start", y="tempC",
    color="session" if is_compare else "zone",
    category_orders={
        "session": session_order,
        "zone":    [z for z in ZONE_COLOURS if z in headline_df["zone"].unique()],
    },
    color_discrete_map=SESSION_COLOURS if is_compare else ZONE_COLOURS,
    symbol="zone" if is_compare else None,
    opacity=0.55,
    labels={
        "minute_from_start": "Minuten sinds start van de sessie",
        "tempC": "Temperatuur (°C)",
        "session": "Sessie",
        "zone": "Zone",
    },
)
fig_headline.update_traces(marker=dict(size=7))
fig_headline.update_layout(
    height=380, margin=dict(l=10, r=10, t=10, b=10),
    legend_title_text="",
    hovermode="closest",
)

# Voeg KNMI omgevingstemperatuur als horizontale referentielijn toe per sessie.
# Dit maakt de sensor-vs-omgeving offset visueel direct af te lezen.
for s in session_order:
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
        "gedocumenteerd voor reproduceerbaarheid. Vul `SESSION_METADATA` in "
        "`data_loader.py` aan voor toekomstige metingen."
    )
    meta_cols = st.columns(len(session_order))
    for col_st, s in zip(meta_cols, session_order):
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
        "tempC":        ("Temperatuur (°C)",        "#ef4444"),
        "pressure_hPa": ("Luchtdruk (hPa)",         "#3b82f6"),
        "light_lux":    ("Licht (lux)",             "#eab308"),
        "mq_raw":       ("MQ gassensor (ruw)",      "#10b981"),
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
                df, x="minute_from_start", y=col,
                color="session",
                category_orders={"session": session_order},
                color_discrete_map=SESSION_COLOURS,
                title=label,
                labels={"minute_from_start": "Minuten sinds start"},
            )
        else:
            fig = px.line(df, x="timestamp", y=col, title=label)
            fig.update_traces(line=dict(color=sensor_color, width=2))

        fig.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis_title=("Minuten sinds start" if is_compare else None),
            yaxis_title=label,
            legend_title_text="" if is_compare else None,
        )
        if col == "light_lux" and log_light:
            fig.update_yaxes(type="log")
        st.plotly_chart(fig, use_container_width=True)


# ---- GPS-route -----------------------------------------------------------
with tab_map:
    st.subheader("GPS-route")

    gps = df.dropna(subset=["lat_dec", "lon_dec"]).reset_index(drop=True)

    if gps.empty:
        st.info("Geen GPS-fixes beschikbaar in dit venster.")
    else:
        # --- Route-statistieken ------------------------------------------
        if is_compare:
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
        c1, c2 = st.columns([2, 1])
        if is_compare:
            colour_label = "Sessie"
            colour_col   = "session"
            c1.info("In vergelijkmodus is de kleur ingesteld op **sessie**.")
        else:
            colour_options = {
                "Tijdsverloop":                          "_time_idx",
                "Temperatuur-afwijking (Δ van gem.)":    "tempC_anomaly",
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

        map_style_label = c2.selectbox("Kaartstijl",
                                       list(MAP_STYLES.keys()), index=0)
        map_style = MAP_STYLES[map_style_label]

        gps["_time_idx"] = np.arange(len(gps))

        # --- Bouw de kaart op (MapLibre-gebaseerde Scattermap) -----------
        fig = go.Figure()

        if is_compare:
            for s in session_order:
                ssub = gps[gps["session"] == s]
                if ssub.empty:
                    continue
                short_name = s.split(" - ")[0]

                fig.add_trace(go.Scattermap(
                    lat=ssub["lat_dec"], lon=ssub["lon_dec"],
                    mode="lines",
                    line=dict(width=3, color=SESSION_COLOURS[s]),
                    name=f"{s} — route",
                    hoverinfo="skip",
                    legendgroup=s,
                ))
                fig.add_trace(go.Scattermap(
                    lat=ssub["lat_dec"], lon=ssub["lon_dec"],
                    mode="markers",
                    marker=dict(size=7, color=SESSION_COLOURS[s]),
                    name=f"{s} — metingen",
                    hovertext=ssub["timestamp"].dt.strftime("%H:%M:%S")
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

        # Auto-zoom om de route te omsluiten
        center_lat = gps["lat_dec"].mean()
        center_lon = gps["lon_dec"].mean()
        span = max(
            gps["lat_dec"].max() - gps["lat_dec"].min(),
            gps["lon_dec"].max() - gps["lon_dec"].min(),
        )
        zoom = 15 if span < 0.01 else 14 if span < 0.03 else 13

        show_legend = is_compare or (not is_compare and colour_col == "zone")
        # Scattermap (MapLibre) gebruikt `map=` in plaats van `mapbox=`.
        # De ingebouwde zoom-/pan-controls zijn responsiever dan de oude
        # Mapbox-versie en vereisen geen Mapbox-token.
        fig.update_layout(
            map=dict(style=map_style,
                     center=dict(lat=center_lat, lon=center_lon),
                     zoom=zoom),
            height=620,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=show_legend,
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
            if is_compare:
                speed_fig = px.line(
                    gps, x="minute_from_start", y="speed_kmh",
                    color="session",
                    category_orders={"session": session_order},
                    color_discrete_map=SESSION_COLOURS,
                    labels={"speed_kmh": "Snelheid (km/h)",
                            "minute_from_start": "Minuten sinds start"},
                )
            else:
                speed_fig = px.line(
                    gps, x="timestamp", y="speed_kmh",
                    labels={"speed_kmh": "Snelheid (km/h)", "timestamp": "Tijd"},
                )
                speed_fig.update_traces(line=dict(color="#3b82f6", width=2))
            speed_fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10),
                                    legend_title_text="")
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
            "Sarphatipark) of *Onderweg* als hij buiten alle drie valt. Pas de "
            "`ZONES`-dict in `app.py` aan om grenzen te verfijnen."
        )

    c1, c2 = st.columns(2)
    exclude_transit = c1.toggle("Sluit metingen 'Onderweg' uit", value=True,
                                help="Verberg GPS-fixes die niet in een van de 3 zones vallen")
    only_stationary = c2.toggle("Alleen stilstaande metingen (< 0.5 km/h)", value=False,
                                help="Sensor heeft ~15s nodig om te stabiliseren tijdens lopen — "
                                     "stilstaande metingen zijn betrouwbaarder")

    z = df.copy()
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
        if is_compare:
            fig_box = px.box(
                z, x="zone", y="tempC", color="session",
                category_orders={"zone": zone_order, "session": session_order},
                color_discrete_map=SESSION_COLOURS,
                points="all",
                labels={"tempC": "Temperatuur (°C)", "zone": "", "session": ""},
            )
            fig_box.update_layout(boxmode="group", height=420,
                                  margin=dict(l=10, r=10, t=10, b=10),
                                  legend_title_text="")
        else:
            fig_box = px.box(
                z, x="zone", y="tempC", color="zone",
                category_orders={"zone": zone_order},
                color_discrete_map=ZONE_COLOURS,
                points="all",
                labels={"tempC": "Temperatuur (°C)", "zone": ""},
            )
            fig_box.update_layout(showlegend=False, height=400,
                                  margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_box, use_container_width=True)

        if is_compare:
            st.caption(
                "Als dezelfde zone-volgorde geldt over beide sessies (bv. Frans Halsbuurt "
                "het warmst in beide, Sarphatipark het koelst in beide), is je ruimtelijke "
                "bevinding robuust voor de wijziging in methodologie."
            )

        # --- Statistische analyse: ANOVA + Tukey HSD ---------------------
        st.markdown("### 📊 Statistische analyse")
        st.caption(
            "Eén-weg ANOVA toetst of de gemiddelde temperatuur significant "
            "verschilt tussen zones. Een **lage p-waarde** (<0.05) betekent: de "
            "verschillen tussen zones zijn groter dan we door toeval zouden "
            "verwachten. **η² (eta-squared)** is de effectgrootte — welk deel "
            "van de variantie wordt verklaard door zone. Cohen-richtlijnen: "
            "0.01 klein, 0.06 middel, 0.14 groot."
        )

        if is_compare:
            anova_cols = st.columns(len(session_order))
            for col_st, s in zip(anova_cols, session_order):
                with col_st:
                    st.markdown(f"**{s}**")
                    sub_anova = z[z["session"] == s]
                    result = run_zone_anova(sub_anova)
                    if result is None:
                        st.info("Te weinig data voor ANOVA in deze sessie.")
                        continue

                    sig_emoji = "✅" if result["p"] < 0.05 else "⚠️"
                    sig_label = "significant" if result["p"] < 0.05 else "niet significant"
                    p_text = f"{result['p']:.4f}" if result['p'] >= 0.0001 else "<0.0001"

                    st.markdown(
                        f"- **F**({result['n_groups']-1}, "
                        f"{result['n_samples']-result['n_groups']}) "
                        f"= {result['F']:.2f}\n"
                        f"- **p** = {p_text} {sig_emoji} ({sig_label})\n"
                        f"- **η²** = {result['eta_squared']:.3f} "
                        f"({interpret_eta_squared(result['eta_squared'])} effect)\n"
                        f"- **N** = {result['n_samples']} metingen over "
                        f"{result['n_groups']} zones"
                    )

                    # Tukey HSD alleen tonen als ANOVA significant
                    if result["p"] < 0.05:
                        tukey = run_tukey_hsd(sub_anova)
                        if tukey is not None:
                            with st.expander("Tukey HSD post-hoc (paarsgewijze vergelijkingen)"):
                                st.dataframe(tukey, use_container_width=True,
                                             hide_index=True)
                                st.caption(
                                    "`reject=True` betekent dat het paar zones "
                                    "significant verschilt (na correctie voor "
                                    "meervoudig toetsen)."
                                )
        else:
            result = run_zone_anova(z)
            if result is None:
                st.info("Te weinig data voor ANOVA met de huidige filters.")
            else:
                sig_emoji = "✅" if result["p"] < 0.05 else "⚠️"
                sig_label = "significant" if result["p"] < 0.05 else "niet significant"
                p_text = f"{result['p']:.4f}" if result['p'] >= 0.0001 else "<0.0001"

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("F-statistiek",
                          f"{result['F']:.2f}",
                          help=f"df_tussen = {result['n_groups']-1}, "
                               f"df_binnen = {result['n_samples']-result['n_groups']}")
                c2.metric("p-waarde", p_text,
                          delta=sig_label, delta_color="off")
                c3.metric("η² (effectgrootte)",
                          f"{result['eta_squared']:.3f}",
                          delta=interpret_eta_squared(result['eta_squared']),
                          delta_color="off")
                c4.metric("N", f"{result['n_samples']}")

                if result["p"] < 0.05:
                    tukey = run_tukey_hsd(z)
                    if tukey is not None:
                        with st.expander("Tukey HSD post-hoc (paarsgewijze vergelijkingen)"):
                            st.dataframe(tukey, use_container_width=True, hide_index=True)
                            st.caption(
                                "`reject=True` betekent dat het paar zones "
                                "significant verschilt (na correctie voor "
                                "meervoudig toetsen)."
                            )

        st.caption(
            "⚠️ **Caveat**: ANOVA veronderstelt onafhankelijke metingen en "
            "ongeveer-normaal verdeelde residuen. Onze metingen zijn "
            "auto-gecorreleerd (opeenvolgende sensorwaarden hangen samen) en "
            "verstrengeld met tijd-van-dag (zones zijn op verschillende "
            "momenten bezocht). Behandel deze toetsen als richtinggevend, niet "
            "als definitief bewijs. Met meer runs wordt een mixed-effects model "
            "met sessie als random effect en tijd als covariaat passender."
        )

        # --- Licht vs Temperatuur scatter --------------------------------
        st.markdown("**Licht vs temperatuur**")
        if is_compare:
            fig_sc = px.scatter(
                z.dropna(subset=["light_lux", "tempC"]),
                x="light_lux", y="tempC",
                color="session",
                facet_col="zone",
                category_orders={"zone": zone_order, "session": session_order},
                color_discrete_map=SESSION_COLOURS,
                log_x=True, opacity=0.7,
                labels={"light_lux": "Licht (lux, log)", "tempC": "Temperatuur (°C)"},
            )
        else:
            fig_sc = px.scatter(
                z.dropna(subset=["light_lux", "tempC"]),
                x="light_lux", y="tempC", color="zone",
                category_orders={"zone": zone_order},
                color_discrete_map=ZONE_COLOURS,
                log_x=True, opacity=0.7,
                labels={"light_lux": "Licht (lux, log)", "tempC": "Temperatuur (°C)"},
            )
        fig_sc.update_traces(marker=dict(size=8))
        fig_sc.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10),
                             legend_title_text="")
        st.plotly_chart(fig_sc, use_container_width=True)

        # --- Tijd-van-dag confounder check -------------------------------
        st.markdown("**Wanneer is elke zone bezocht?** *(tijd-van-dag controle)*")
        if is_compare:
            timeline = (z.assign(minute=z["minute_from_start"].round())
                         .groupby(["minute", "zone", "session"])
                         .size().reset_index(name="metingen"))
            fig_tl = px.scatter(
                timeline, x="minute", y="zone",
                size="metingen", color="zone",
                facet_row="session",
                category_orders={"zone": zone_order, "session": session_order},
                color_discrete_map=ZONE_COLOURS,
                labels={"minute": "Minuten sinds start", "zone": ""},
            )
            fig_tl.update_layout(showlegend=False, height=320,
                                 margin=dict(l=10, r=10, t=30, b=10))
        else:
            timeline = (z.assign(minute=z["timestamp"].dt.floor("1min"))
                         .groupby(["minute", "zone"]).size().reset_index(name="metingen"))
            fig_tl = px.scatter(
                timeline, x="minute", y="zone", size="metingen", color="zone",
                category_orders={"zone": zone_order},
                color_discrete_map=ZONE_COLOURS,
                labels={"minute": "Tijd", "zone": ""},
            )
            fig_tl.update_layout(showlegend=False, height=220,
                                 margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_tl, use_container_width=True)
        st.caption(
            "Als zones in volgorde bezocht zijn (niet door elkaar), kan een deel van het "
            "temperatuurverschil tijd-van-dag zijn in plaats van locatie. Vermeld dit "
            "voorbehoud in je discussie."
        )


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

    if is_compare:
        st.markdown("**Correlatiematrix per sessie**")
        cols_st = st.columns(len(session_order))
        for col_st, s in zip(cols_st, session_order):
            sub = df[df["session"] == s]
            corr = sub[sensor_cols].corr()
            corr.index = corr.index.map(sensor_labels_nl)
            corr.columns = corr.columns.map(sensor_labels_nl)
            fig = px.imshow(
                corr, text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1, aspect="auto",
                title=s,
            )
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
            col_st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Als een correlatie sterk verschilt tussen sessies, is dat een aanwijzing dat "
            "de deksel-wijziging de sensor-kruisgevoeligheid heeft beïnvloed (bv. "
            "licht→temp koppeling)."
        )
    else:
        corr = df[sensor_cols].corr()
        corr.index = corr.index.map(sensor_labels_nl)
        corr.columns = corr.columns.map(sensor_labels_nl)
        fig = px.imshow(
            corr, text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1, aspect="auto",
        )
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Paar-scatter**")
    c1, c2 = st.columns(2)
    x_axis = c1.selectbox("X-as", sensor_cols, index=0,
                          format_func=lambda c: sensor_labels_nl[c])
    y_axis = c2.selectbox("Y-as", sensor_cols, index=2,
                          format_func=lambda c: sensor_labels_nl[c])

    if is_compare:
        fig2 = px.scatter(
            df, x=x_axis, y=y_axis,
            color="session",
            category_orders={"session": session_order},
            color_discrete_map=SESSION_COLOURS,
            trendline="ols",
            opacity=0.6, height=420,
            labels={x_axis: sensor_labels_nl[x_axis], y_axis: sensor_labels_nl[y_axis]},
        )
    else:
        fig2 = px.scatter(
            df, x=x_axis, y=y_axis,
            trendline="ols",
            opacity=0.6, height=420,
            labels={x_axis: sensor_labels_nl[x_axis], y_axis: sensor_labels_nl[y_axis]},
        )
    fig2.update_layout(margin=dict(l=10, r=10, t=10, b=10), legend_title_text="")
    st.plotly_chart(fig2, use_container_width=True)


# ---- Ruwe data -----------------------------------------------------------
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