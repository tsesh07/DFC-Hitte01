"""
Arduino Sensor Logger Dashboard
================================
Reads the cleaned multi-session dataframe via data_loader.load_data() and
visualises GPS track + environmental sensor readings, with per-tab support
for comparing Day 1 (lid closed) vs Day 2 (lid open).

Run with:  streamlit run app.py
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_loader import load_data  # cleaned, multi-session dataframe lives here

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Sensor Logger Dashboard",
    page_icon="📡",
    layout="wide",
)

# --------------------------------------------------------------------------
# Coordinate Reference Systems
# --------------------------------------------------------------------------
# WGS84 (EPSG:4326)  — the standard "GPS" CRS, lat/lon in degrees.
#                      What the Arduino's GPS module reports.
# RD New (EPSG:28992) — Dutch national grid, coordinates in metres.
#                      Used for any operation where we need real-world distances
#                      (buffering zones, computing 'radius in metres', etc.).
WGS84 = "EPSG:4326"
RD_NEW = "EPSG:28992"

# --------------------------------------------------------------------------
# Zone definitions (Amsterdam) — edit centres / radii to fit your transect
# --------------------------------------------------------------------------
# Each zone is defined as a centre point in WGS84 (lat/lon) plus a radius in
# METRES. We buffer the centre point in EPSG:28992 to get a circular polygon
# that has the same metric radius in every direction — unlike a bounding box
# in lat/lon, which would be stretched east-west at this latitude.
ZONES = {
    "Museumplein":     {"lon": 4.8810, "lat": 52.3580, "radius_m": 200, "surface": "paved"},
    "Frans Halsbuurt": {"lon": 4.8920, "lat": 52.3563, "radius_m": 150, "surface": "paved"},
    "Sarphatipark":    {"lon": 4.8950, "lat": 52.3540, "radius_m": 120, "surface": "tree-canopy"},
}
ZONE_COLOURS = {
    "Museumplein":     "#ef4444",
    "Frans Halsbuurt": "#3b82f6",
    "Sarphatipark":    "#10b981",
    "Transit":         "#9ca3af",
}

# Distinct, accessible colours for session comparison. Blue/orange is a
# colour-blind-safe pairing and contrasts well on both light and dark map tiles.
SESSION_COLOURS = {
    "Day 1 - lid closed": "#2563eb",  # blue
    "Day 2 - lid open":   "#f97316",  # orange
}


# --------------------------------------------------------------------------
# Geo helpers
# --------------------------------------------------------------------------
@st.cache_data
def build_zones_gdf() -> gpd.GeoDataFrame:
    """
    Build a GeoDataFrame of zone polygons.

    Pipeline:
      1. Create centre Points in WGS84 (lat/lon, degrees).
      2. Reproject to RD New (metres) so that .buffer() works in metres.
      3. Replace each centre Point with a circular buffer of the requested radius.

    Returns a GeoDataFrame in EPSG:28992 with columns: zone, surface, geometry.
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
    """
    Spatial join: for each sample, find which zone polygon it falls inside.

    `sjoin` ('spatial join') matches rows from one GeoDataFrame to another
    based on a geometric predicate ('within', 'intersects', 'contains', ...).

    Steps:
      1. Build a GeoDataFrame of sample Points in WGS84.
      2. Reproject to RD New so they live in the same CRS as the zone polygons.
      3. sjoin with predicate='within' to test point-in-polygon.
      4. Fill missing matches (samples outside all zones) with 'Transit'.
    """
    valid = df["lat_dec"].notna() & df["lon_dec"].notna()

    points = gpd.GeoDataFrame(
        df.loc[valid, ["lat_dec", "lon_dec"]].copy(),
        geometry=gpd.points_from_xy(df.loc[valid, "lon_dec"], df.loc[valid, "lat_dec"]),
        crs=WGS84,
    ).to_crs(RD_NEW)

    joined = gpd.sjoin(points, zones_gdf[["zone", "geometry"]],
                       how="left", predicate="within")

    result = pd.Series("Transit", index=df.index, dtype="object")
    result.loc[valid] = joined["zone"].fillna("Transit").values
    return result


# --------------------------------------------------------------------------
# Track / sensor helpers
# --------------------------------------------------------------------------
def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres between two points (vectorised)."""
    R = 6_371_000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def add_track_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute step distance (m), cumulative distance (m), and speed (m/s, km/h).

    Calculated PER SESSION so we don't end up with a phantom 'step' from the
    last point of Day 1 to the first point of Day 2.
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
    """
    Add `minute_from_start`: minutes since the first sample of each session.

    Lets us overlay sessions that happened on different calendar days on a
    common x-axis in the time-series tab.
    """
    df = df.copy()
    df["minute_from_start"] = df.groupby("session")["timestamp"].transform(
        lambda x: (x - x.min()).dt.total_seconds() / 60
    )
    return df


def clean_sensor_data(df: pd.DataFrame, drop_glitches: bool) -> pd.DataFrame:
    """Optionally mask obvious sensor glitches.

    Raw values arrive untouched from data_loader.load_data(); this is the
    one place where we (optionally) blank out unphysical readings for plotting.
    """
    df = df.copy()
    if drop_glitches:
        # BMP280 needs ~1s warmup — readings well below sea level are wrong
        df.loc[df["pressure_hPa"] < 900, "pressure_hPa"] = np.nan
        # Sentinel value from the photoresistor before first valid read
        df.loc[df["light_lux"] < 0, "light_lux"] = np.nan
        # Massive negative or unrealistically low tempC = sensor disconnect.
        # 5 °C is a safe lower bound for outdoor measurements in Amsterdam in spring.
        # Day 2 also has high-side glitches (the +144 °C spikes), clamp upper too.
        df.loc[(df["tempC"] < 5) | (df["tempC"] > 50), "tempC"] = np.nan
    return df


# --------------------------------------------------------------------------
# Sidebar — data source and filtering
# --------------------------------------------------------------------------
st.sidebar.header("⚙️ Data settings")

try:
    raw = load_data()
except FileNotFoundError as e:
    st.error(f"Could not load raw data: {e}\n\n"
             "Expected at `../data/raw/DATA.CSV` relative to `app.py`.")
    st.stop()

available_sessions = list(raw["session"].unique())

sessions = st.sidebar.multiselect(
    "Sessions to display",
    options=available_sessions,
    default=available_sessions,
    help="Select one for full single-session view, or both to compare.",
)

if not sessions:
    st.warning("Pick at least one session in the sidebar.")
    st.stop()

drop_glitches = st.sidebar.toggle(
    "Mask sensor glitches", value=True,
    help="Hide BMP280 warmup pressure values, −1 lux sentinels, and "
         "out-of-range temperature spikes (sensor disconnects).",
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Loaded data**")
for s in available_sessions:
    n = (raw["session"] == s).sum()
    st.sidebar.markdown(f"- {s}: **{n:,}** rows")

# --------------------------------------------------------------------------
# Filter & enrich the dataframe
# --------------------------------------------------------------------------
df = raw[raw["session"].isin(sessions)].copy()
df = clean_sensor_data(df, drop_glitches)
df = add_track_metrics(df)
df = add_minutes_from_start(df)

zones_gdf = build_zones_gdf()
df["zone"] = assign_zones_via_sjoin(df, zones_gdf)
df["tempC_anomaly"] = df["tempC"] - df["tempC"].mean()

is_compare = len(sessions) > 1
session_order = [s for s in available_sessions if s in sessions]  # stable display order

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("📡 Arduino Sensor Logger Dashboard")
if is_compare:
    st.caption(
        f"Comparing **{len(sessions)} sessions** "
        f"({', '.join(sessions)}) — {len(df):,} samples total."
    )
else:
    only = sessions[0]
    sub = df[df["session"] == only]
    st.caption(
        f"**{only}** — {len(sub):,} samples from "
        f"{sub['timestamp'].min():%Y-%m-%d %H:%M:%S} to {sub['timestamp'].max():%H:%M:%S}."
    )

if df.empty:
    st.warning("No rows match the current filters.")
    st.stop()

# --------------------------------------------------------------------------
# Top-line KPIs
# --------------------------------------------------------------------------
if is_compare:
    # Side-by-side comparison table — one column per session
    kpi_rows = {}
    for s in session_order:
        sub = df[df["session"] == s]
        duration = (sub["timestamp"].max() - sub["timestamp"].min()).total_seconds() / 60
        kpi_rows[s] = {
            "Samples":              f"{len(sub):,}",
            "Duration (min)":       f"{duration:.1f}",
            "Avg temperature (°C)": f"{sub['tempC'].mean():.1f}",
            "Avg pressure (hPa)":   f"{sub['pressure_hPa'].mean():.1f}",
            "Peak light (lx)":      f"{sub['light_lux'].max():,.0f}",
        }
    kpi_df = pd.DataFrame(kpi_rows)
    st.dataframe(kpi_df, use_container_width=True)
else:
    sub = df  # only one session selected
    duration = (sub["timestamp"].max() - sub["timestamp"].min()).total_seconds() / 60
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Samples", f"{len(sub):,}")
    c2.metric("Duration", f"{duration:.1f} min")
    c3.metric("Avg temperature", f"{sub['tempC'].mean():.1f} °C")
    c4.metric("Avg pressure", f"{sub['pressure_hPa'].mean():.1f} hPa")
    c5.metric("Peak light", f"{sub['light_lux'].max():,.0f} lx")

st.divider()

# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
tab_time, tab_map, tab_zones, tab_corr, tab_data = st.tabs(
    ["📈 Time series", "🗺️ GPS track", "🏛️ Zone analysis", "🔗 Correlations", "📋 Raw data"]
)

# ---- Time series ---------------------------------------------------------
with tab_time:
    st.subheader("Sensor readings over time")
    if is_compare:
        st.caption(
            "X-axis is **minutes from each session's start** so the two walks "
            "overlay on a common timeline despite happening on different days."
        )

    sensor_meta = {
        "tempC":        ("Temperature (°C)",     "#ef4444"),
        "pressure_hPa": ("Pressure (hPa)",       "#3b82f6"),
        "light_lux":    ("Light (lux)",          "#eab308"),
        "mq_raw":       ("MQ gas sensor (raw)",  "#10b981"),
    }

    selected = st.multiselect(
        "Sensors to display",
        options=list(sensor_meta.keys()),
        default=list(sensor_meta.keys()),
        format_func=lambda c: sensor_meta[c][0],
    )

    log_light = st.checkbox("Use log scale for light", value=True,
                            help="Outdoor sun and indoor readings differ by 1000×")

    for col in selected:
        label, sensor_color = sensor_meta[col]
        if is_compare:
            fig = px.line(
                df, x="minute_from_start", y=col,
                color="session",
                category_orders={"session": session_order},
                color_discrete_map=SESSION_COLOURS,
                title=label,
                labels={"minute_from_start": "Minutes from session start"},
            )
        else:
            fig = px.line(df, x="timestamp", y=col, title=label)
            fig.update_traces(line=dict(color=sensor_color, width=2))

        fig.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis_title=("Minutes from start" if is_compare else None),
            yaxis_title=label,
            legend_title_text="" if is_compare else None,
        )
        if col == "light_lux" and log_light:
            fig.update_yaxes(type="log")
        st.plotly_chart(fig, use_container_width=True)

# ---- Map -----------------------------------------------------------------
with tab_map:
    st.subheader("GPS track")

    gps = df.dropna(subset=["lat_dec", "lon_dec"]).reset_index(drop=True)

    if gps.empty:
        st.info("No GPS fixes available in this window.")
    else:
        # --- Route metrics -------------------------------------------
        if is_compare:
            metrics_rows = {}
            for s in session_order:
                ssub = gps[gps["session"] == s]
                if ssub.empty:
                    continue
                dur = (ssub["timestamp"].max() - ssub["timestamp"].min()).total_seconds() / 60
                metrics_rows[s] = {
                    "Distance (km)":       f"{ssub['step_m'].sum() / 1000:.2f}",
                    "Avg speed (km/h)":    f"{ssub['speed_kmh'].mean():.2f}",
                    "Peak speed p95":      f"{ssub['speed_kmh'].quantile(0.95):.2f}",
                    "Moving time (min)":   f"{dur:.1f}",
                }
            st.dataframe(pd.DataFrame(metrics_rows), use_container_width=True)
        else:
            total_dist_m = gps["step_m"].sum(skipna=True)
            avg_speed   = gps["speed_kmh"].mean(skipna=True)
            peak_speed  = gps["speed_kmh"].quantile(0.95)
            track_min   = (gps["timestamp"].max() - gps["timestamp"].min()).total_seconds() / 60

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Distance walked", f"{total_dist_m/1000:.2f} km")
            m2.metric("Avg speed",       f"{avg_speed:.2f} km/h")
            m3.metric("Peak speed (p95)", f"{peak_speed:.2f} km/h")
            m4.metric("Moving time",     f"{track_min:.1f} min")

        # --- Display options -----------------------------------------
        c1, c2 = st.columns([2, 1])
        if is_compare:
            # In compare mode we always colour by session — overlaying two
            # routes with continuous colourings is unreadable.
            colour_label = "Session"
            colour_col   = "session"
            c1.info("Colouring is set to **session** in compare mode.")
        else:
            colour_options = {
                "Time progression":                  "_time_idx",
                "Temperature anomaly (Δ from mean)": "tempC_anomaly",
                "Speed (km/h)":                      "speed_kmh",
                "Temperature (°C)":                  "tempC",
                "Pressure (hPa)":                    "pressure_hPa",
                "Light (lux)":                       "light_lux",
                "MQ gas (raw)":                      "mq_raw",
                "Zone":                              "zone",
            }
            colour_label = c1.selectbox("Colour route by", list(colour_options.keys()),
                                        index=1)
            colour_col = colour_options[colour_label]

        map_style = c2.selectbox("Map style",
                                 ["open-street-map", "carto-positron", "carto-darkmatter"],
                                 index=0)

        gps["_time_idx"] = np.arange(len(gps))

        # --- Build the map -------------------------------------------
        fig = go.Figure()

        if is_compare:
            # One route + points trace per session, coloured by session.
            # Each session also gets its own Start/End markers (green/red), labelled
            # with the session's short name so it's clear which is which.
            for s in session_order:
                ssub = gps[gps["session"] == s]
                if ssub.empty:
                    continue
                short_name = s.split(" - ")[0]  # "Day 1" / "Day 2"

                fig.add_trace(go.Scattermapbox(
                    lat=ssub["lat_dec"], lon=ssub["lon_dec"],
                    mode="lines",
                    line=dict(width=3, color=SESSION_COLOURS[s]),
                    name=f"{s} — route",
                    hoverinfo="skip",
                    legendgroup=s,
                ))
                fig.add_trace(go.Scattermapbox(
                    lat=ssub["lat_dec"], lon=ssub["lon_dec"],
                    mode="markers",
                    marker=dict(size=7, color=SESSION_COLOURS[s]),
                    name=f"{s} — samples",
                    hovertext=ssub["timestamp"].dt.strftime("%H:%M:%S")
                              + " — " + ssub["tempC"].round(1).astype(str) + " °C",
                    hoverinfo="text",
                    legendgroup=s,
                ))

                start_pt = ssub.iloc[0]
                end_pt   = ssub.iloc[-1]
                fig.add_trace(go.Scattermapbox(
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
                fig.add_trace(go.Scattermapbox(
                    lat=[end_pt["lat_dec"]], lon=[end_pt["lon_dec"]],
                    mode="markers+text",
                    marker=dict(size=16, color="#ef4444"),
                    text=[f"{short_name} end"], textposition="top right",
                    textfont=dict(size=12, color="#ef4444"),
                    name=f"{short_name} end",
                    hovertext=f"{s} end: {end_pt['timestamp']:%H:%M:%S}",
                    hoverinfo="text",
                    legendgroup=s,
                ))
        else:
            # Single-session view with full continuous-coloring options.
            fig.add_trace(go.Scattermapbox(
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
                    fig.add_trace(go.Scattermapbox(
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

                fig.add_trace(go.Scattermapbox(
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
                    name="Samples",
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
                        "Speed: %{customdata[1]:.2f} km/h<br>"
                        "Distance from start: %{customdata[2]:.0f} m<br>"
                        "Temp: %{customdata[3]:.1f} °C<br>"
                        "Pressure: %{customdata[4]:.1f} hPa<br>"
                        "Light: %{customdata[5]:.0f} lux<br>"
                        "MQ: %{customdata[6]:.0f}<extra></extra>"
                    ),
                ))

            # Start / End markers (only in single-session view; doubling them
            # for two sessions clutters the map).
            start_pt = gps.iloc[0]
            end_pt   = gps.iloc[-1]
            fig.add_trace(go.Scattermapbox(
                lat=[start_pt["lat_dec"]], lon=[start_pt["lon_dec"]],
                mode="markers+text",
                marker=dict(size=18, color="#10b981"),
                text=["Start"], textposition="top right",
                textfont=dict(size=14, color="#10b981"),
                name="Start",
                hovertext=f"Start: {start_pt['timestamp']:%H:%M:%S}",
                hoverinfo="text",
            ))
            fig.add_trace(go.Scattermapbox(
                lat=[end_pt["lat_dec"]], lon=[end_pt["lon_dec"]],
                mode="markers+text",
                marker=dict(size=18, color="#ef4444"),
                text=["End"], textposition="top right",
                textfont=dict(size=14, color="#ef4444"),
                name="End",
                hovertext=f"End: {end_pt['timestamp']:%H:%M:%S}",
                hoverinfo="text",
            ))

        # Auto-zoom to fit the track
        center_lat = gps["lat_dec"].mean()
        center_lon = gps["lon_dec"].mean()
        span = max(
            gps["lat_dec"].max() - gps["lat_dec"].min(),
            gps["lon_dec"].max() - gps["lon_dec"].min(),
        )
        zoom = 15 if span < 0.01 else 14 if span < 0.03 else 13

        show_legend = is_compare or (not is_compare and colour_col == "zone")
        fig.update_layout(
            mapbox=dict(style=map_style, center=dict(lat=center_lat, lon=center_lon), zoom=zoom),
            height=600,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=show_legend,
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Speed-over-time chart -----------------------------------
        with st.expander("📊 Speed over time", expanded=False):
            if is_compare:
                speed_fig = px.line(
                    gps, x="minute_from_start", y="speed_kmh",
                    color="session",
                    category_orders={"session": session_order},
                    color_discrete_map=SESSION_COLOURS,
                    labels={"speed_kmh": "Speed (km/h)",
                            "minute_from_start": "Minutes from session start"},
                )
            else:
                speed_fig = px.line(
                    gps, x="timestamp", y="speed_kmh",
                    labels={"speed_kmh": "Speed (km/h)", "timestamp": "Time"},
                )
                speed_fig.update_traces(line=dict(color="#3b82f6", width=2))
            speed_fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10),
                                    legend_title_text="")
            st.plotly_chart(speed_fig, use_container_width=True)
            st.caption(
                "Speeds are derived from haversine distance between consecutive "
                "GPS fixes (~5s apart). Walking pace is typically 4–6 km/h; anything "
                "above ~10 km/h is usually GPS jitter rather than real movement."
            )

# ---- Zone analysis -------------------------------------------------------
with tab_zones:
    st.subheader("Micro-climate comparison across zones")
    if is_compare:
        st.caption(
            "Each GPS fix is assigned to a zone and a session. The headline "
            "thesis question — *does the spatial pattern from Day 1 replicate on "
            "Day 2?* — lives in this tab."
        )
    else:
        st.caption(
            "Each GPS fix is assigned to a zone (Museumplein, Frans Halsbuurt, "
            "Sarphatipark) or *Transit* if it falls outside all three. Edit the "
            "`ZONES` dict in `app.py` to refine boundaries."
        )

    c1, c2 = st.columns(2)
    exclude_transit = c1.toggle("Exclude transit samples", value=True,
                                help="Hide GPS fixes that aren't inside any of the 3 zones")
    only_stationary = c2.toggle("Only stationary samples (< 0.5 km/h)", value=False,
                                help="Sensor needs ~15s to equilibrate while walking — "
                                     "stationary readings are more reliable")

    z = df.copy()
    if exclude_transit:
        z = z[z["zone"] != "Transit"]
    if only_stationary:
        z = z[z["speed_kmh"].fillna(0) < 0.5]

    if len(z) == 0:
        st.warning("No samples match the current filters.")
    else:
        zone_order = [z_ for z_ in ZONE_COLOURS if z_ in z["zone"].unique()]

        # --- Summary table -----------------------------------------------
        group_cols = ["zone", "session"] if is_compare else ["zone"]
        summary = (z.groupby(group_cols)
                    .agg(samples=("tempC", "count"),
                         mean_temp=("tempC", "mean"),
                         std_temp=("tempC", "std"),
                         min_temp=("tempC", "min"),
                         max_temp=("tempC", "max"),
                         median_light=("light_lux", "median"),
                         max_light=("light_lux", "max"),
                         mean_mq=("mq_raw", "mean"))
                    .round(2))
        st.markdown("**Summary by zone**" + (" × session" if is_compare else ""))
        st.dataframe(summary, use_container_width=True)

        # --- Boxplot of temperature -------------------------------------
        st.markdown("**Temperature distribution by zone**")
        if is_compare:
            fig_box = px.box(
                z, x="zone", y="tempC", color="session",
                category_orders={"zone": zone_order, "session": session_order},
                color_discrete_map=SESSION_COLOURS,
                points="all",
                labels={"tempC": "Temperature (°C)", "zone": "", "session": ""},
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
                labels={"tempC": "Temperature (°C)", "zone": ""},
            )
            fig_box.update_layout(showlegend=False, height=400,
                                  margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_box, use_container_width=True)

        if is_compare:
            st.caption(
                "If the same zone ranking holds across both sessions (e.g. "
                "Frans Halsbuurt warmest in both, Sarphatipark coolest in both), "
                "your spatial finding is robust to the lid-on/lid-off methodology change."
            )

        # --- Light vs Temp scatter ---------------------------------------
        st.markdown("**Light vs temperature**")
        if is_compare:
            # Facet by zone, colour by session: lets the viewer see whether
            # the light→temp relationship within a zone is consistent across days.
            fig_sc = px.scatter(
                z.dropna(subset=["light_lux", "tempC"]),
                x="light_lux", y="tempC",
                color="session",
                facet_col="zone",
                category_orders={"zone": zone_order, "session": session_order},
                color_discrete_map=SESSION_COLOURS,
                log_x=True, opacity=0.7,
                labels={"light_lux": "Light (lux, log)", "tempC": "Temperature (°C)"},
            )
        else:
            fig_sc = px.scatter(
                z.dropna(subset=["light_lux", "tempC"]),
                x="light_lux", y="tempC", color="zone",
                category_orders={"zone": zone_order},
                color_discrete_map=ZONE_COLOURS,
                log_x=True, opacity=0.7,
                labels={"light_lux": "Light (lux, log scale)", "tempC": "Temperature (°C)"},
            )
        fig_sc.update_traces(marker=dict(size=8))
        fig_sc.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10),
                             legend_title_text="")
        st.plotly_chart(fig_sc, use_container_width=True)

        # --- Time-of-day exposure ---------------------------------------
        st.markdown("**When was each zone visited?** *(time-of-day confounder check)*")
        if is_compare:
            timeline = (z.assign(minute=z["minute_from_start"].round())
                         .groupby(["minute", "zone", "session"])
                         .size().reset_index(name="samples"))
            fig_tl = px.scatter(
                timeline, x="minute", y="zone",
                size="samples", color="zone",
                facet_row="session",
                category_orders={"zone": zone_order, "session": session_order},
                color_discrete_map=ZONE_COLOURS,
                labels={"minute": "Minutes from session start", "zone": ""},
            )
            fig_tl.update_layout(showlegend=False, height=320,
                                 margin=dict(l=10, r=10, t=30, b=10))
        else:
            timeline = (z.assign(minute=z["timestamp"].dt.floor("1min"))
                         .groupby(["minute", "zone"]).size().reset_index(name="samples"))
            fig_tl = px.scatter(
                timeline, x="minute", y="zone", size="samples", color="zone",
                category_orders={"zone": zone_order},
                color_discrete_map=ZONE_COLOURS,
                labels={"minute": "Time", "zone": ""},
            )
            fig_tl.update_layout(showlegend=False, height=220,
                                 margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_tl, use_container_width=True)
        st.caption(
            "If the zones were visited in sequence (not interleaved), part of the "
            "temperature difference may be time-of-day rather than location. "
            "Mention this caveat in your discussion section."
        )

# ---- Correlations --------------------------------------------------------
with tab_corr:
    st.subheader("How do sensors relate?")

    sensor_cols = ["tempC", "pressure_hPa", "light_lux", "mq_raw"]

    if is_compare:
        st.markdown("**Correlation matrix per session**")
        cols_st = st.columns(len(session_order))
        for col_st, s in zip(cols_st, session_order):
            sub = df[df["session"] == s]
            corr = sub[sensor_cols].corr()
            fig = px.imshow(
                corr, text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1, aspect="auto",
                title=s,
            )
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
            col_st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "If a correlation differs strongly between sessions, that's a hint "
            "the lid change affected sensor cross-sensitivity (e.g. light→temp coupling)."
        )
    else:
        corr = df[sensor_cols].corr()
        fig = px.imshow(
            corr, text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1, aspect="auto",
        )
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Pairwise scatter**")
    c1, c2 = st.columns(2)
    x_axis = c1.selectbox("X axis", sensor_cols, index=0)
    y_axis = c2.selectbox("Y axis", sensor_cols, index=2)

    if is_compare:
        fig2 = px.scatter(
            df, x=x_axis, y=y_axis,
            color="session",
            category_orders={"session": session_order},
            color_discrete_map=SESSION_COLOURS,
            trendline="ols",
            opacity=0.6, height=420,
        )
    else:
        fig2 = px.scatter(
            df, x=x_axis, y=y_axis,
            trendline="ols",
            opacity=0.6, height=420,
        )
    fig2.update_layout(margin=dict(l=10, r=10, t=10, b=10), legend_title_text="")
    st.plotly_chart(fig2, use_container_width=True)

# ---- Raw data ------------------------------------------------------------
with tab_data:
    st.subheader("Filtered records")
    st.dataframe(df, use_container_width=True, height=500)

    csv_bytes = df.to_csv(index=False).encode()
    fname = "sessions_combined.csv" if is_compare else f"{sessions[0].replace(' ', '_')}.csv"
    st.download_button(
        "⬇️ Download filtered CSV",
        data=csv_bytes,
        file_name=fname,
        mime="text/csv",
    )