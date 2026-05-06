"""
Arduino Sensor Logger Dashboard
================================
Reads DATA.CSV from the microSD card, filters out other students' sessions,
and visualises GPS track + environmental sensor readings.

Run with:  streamlit run app.py
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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
# Each zone is defined as a centre point in WGS84 (lat/lon) plus a radius
# in METRES. We buffer the centre point in EPSG:28992 to get a circular
# polygon that has the same metric radius in every direction — unlike a
# bounding box in lat/lon, which would be stretched east-west at this latitude.
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

    # Buffer each centre point by its radius (now in metres because we reprojected)
    gdf["geometry"] = gdf.apply(lambda row: row.geometry.buffer(row["radius_m"]), axis=1)
    return gdf[["zone", "surface", "geometry"]]


def assign_zones_via_sjoin(df: pd.DataFrame, zones_gdf: gpd.GeoDataFrame) -> pd.Series:
    """
    Spatial join: for each sample, find which zone polygon it falls inside.

    This replaces a hand-rolled loop with one of geopandas' core operations:
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

    # Build a result Series aligned to the original df index
    result = pd.Series("Transit", index=df.index, dtype="object")
    result.loc[valid] = joined["zone"].fillna("Transit").values
    return result


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def nmea_to_decimal(value, hemisphere):
    """Convert NMEA DDMM.MMMMM (or DDDMM.MMMMM) to decimal degrees."""
    if pd.isna(value) or pd.isna(hemisphere):
        return np.nan
    degrees = int(value // 100)
    minutes = value - degrees * 100
    decimal = degrees + minutes / 60.0
    if hemisphere in ("S", "W"):
        decimal = -decimal
    return decimal


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["lat_dec"] = df.apply(lambda r: nmea_to_decimal(r["lat"], r["ns"]), axis=1)
    df["lon_dec"] = df.apply(lambda r: nmea_to_decimal(r["lon"], r["ew"]), axis=1)
    return df


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres between two points (vectorised)."""
    R = 6_371_000.0  # Earth radius in metres
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def add_track_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute step distance (m), cumulative distance (m), and speed (m/s, km/h)."""
    df = df.copy()
    df["step_m"] = haversine_m(
        df["lat_dec"].shift(), df["lon_dec"].shift(),
        df["lat_dec"],         df["lon_dec"],
    )
    dt = df["timestamp"].diff().dt.total_seconds()
    df["speed_ms"] = df["step_m"] / dt.replace(0, np.nan)
    df["speed_kmh"] = df["speed_ms"] * 3.6
    df["cum_dist_m"] = df["step_m"].fillna(0).cumsum()
    return df


def clean_sensor_data(df: pd.DataFrame, drop_glitches: bool) -> pd.DataFrame:
    """Optionally mask obvious sensor glitches."""
    df = df.copy()
    if drop_glitches:
        # BMP280 needs ~1 second warmup — readings well below sea level are wrong
        df.loc[df["pressure_hPa"] < 900, "pressure_hPa"] = np.nan
        # Sentinel value from the photoresistor before first valid read
        df.loc[df["light_lux"] < 0, "light_lux"] = np.nan
        # Massive negative or unrealistically low tempC values are sensor disconnects.
        # 5 °C is a safe lower bound for outdoor measurements in Amsterdam in spring.
        df.loc[df["tempC"] < 5, "tempC"] = np.nan
    return df


# --------------------------------------------------------------------------
# Sidebar — data source and filtering
# --------------------------------------------------------------------------
st.sidebar.header("⚙️ Data settings")

csv_path = st.sidebar.text_input("CSV path", value="DATA.CSV")

try:
    raw = load_data(csv_path)
except FileNotFoundError:
    st.error(f"Could not find `{csv_path}`. Put DATA.CSV next to app.py or update the path.")
    st.stop()

st.sidebar.markdown(
    f"**File contains {len(raw):,} rows**  \n"
    f"`gps_time` range: {int(raw['gps_time'].min())} → {int(raw['gps_time'].max())}"
)

st.sidebar.subheader("Your session window")
start_gps = st.sidebar.number_input("Start gps_time (HHMMSS UTC)", value=121155, step=1)
end_gps = st.sidebar.number_input("End gps_time (HHMMSS UTC)", value=130545, step=1)

drop_glitches = st.sidebar.toggle("Mask sensor glitches", value=True,
                                  help="Hide BMP280 warmup pressure values, "
                                       "−1 lux sentinels, and disconnected-sensor temps")

# Apply session filter
mask = raw["gps_time"].between(start_gps, end_gps)
df = raw.loc[mask].reset_index(drop=True)
df = clean_sensor_data(df, drop_glitches)
df = add_track_metrics(df)

# Spatial join: assign each sample to a zone using geopandas
zones_gdf = build_zones_gdf()
df["zone"] = assign_zones_via_sjoin(df, zones_gdf)
df["tempC_anomaly"] = df["tempC"] - df["tempC"].mean()

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("📡 Arduino Sensor Logger Dashboard")
st.caption(
    f"Showing **{len(df):,}** samples "
    f"from {df['timestamp'].min():%Y-%m-%d %H:%M:%S} "
    f"to {df['timestamp'].max():%H:%M:%S} "
    f"(filtered from {len(raw):,} rows on the SD card)."
)

if df.empty:
    st.warning("No rows fall inside the chosen gps_time window.")
    st.stop()

# --------------------------------------------------------------------------
# Top-line KPIs
# --------------------------------------------------------------------------
duration = df["timestamp"].max() - df["timestamp"].min()
mins = duration.total_seconds() / 60

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Samples", f"{len(df):,}")
c2.metric("Duration", f"{mins:.1f} min")
c3.metric("Avg temperature", f"{df['tempC'].mean():.1f} °C")
c4.metric("Avg pressure", f"{df['pressure_hPa'].mean():.1f} hPa")
c5.metric("Peak light", f"{df['light_lux'].max():,.0f} lx")

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
        label, color = sensor_meta[col]
        fig = px.line(df, x="timestamp", y=col, title=label)
        fig.update_traces(line=dict(color=color, width=2))
        fig.update_layout(
            height=280,
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis_title=None,
            yaxis_title=label,
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
        # --- Route metrics ---------------------------------------------
        total_dist_m = gps["step_m"].sum(skipna=True)
        avg_speed = gps["speed_kmh"].mean(skipna=True)
        peak_speed = gps["speed_kmh"].quantile(0.95)  # 95th pct = robust max
        track_minutes = (gps["timestamp"].max() - gps["timestamp"].min()).total_seconds() / 60

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Distance walked", f"{total_dist_m/1000:.2f} km")
        m2.metric("Avg speed",       f"{avg_speed:.2f} km/h")
        m3.metric("Peak speed (p95)", f"{peak_speed:.2f} km/h")
        m4.metric("Moving time",     f"{track_minutes:.1f} min")

        # --- Display options -------------------------------------------
        c1, c2 = st.columns([2, 1])
        colour_options = {
            "Time progression":   "_time_idx",
            "Temperature anomaly (Δ from mean)": "tempC_anomaly",
            "Speed (km/h)":       "speed_kmh",
            "Temperature (°C)":   "tempC",
            "Pressure (hPa)":     "pressure_hPa",
            "Light (lux)":        "light_lux",
            "MQ gas (raw)":       "mq_raw",
            "Zone":               "zone",
        }
        colour_label = c1.selectbox("Colour route by", list(colour_options.keys()),
                                    index=1)  # default to Temperature anomaly
        colour_col = colour_options[colour_label]
        map_style = c2.selectbox("Map style",
                                 ["open-street-map", "carto-positron", "carto-darkmatter"],
                                 index=0)

        # Helper column for time-progression colouring
        gps["_time_idx"] = np.arange(len(gps))

        # --- Build the map ---------------------------------------------
        fig = go.Figure()

        # 1) Route line (drawn underneath so points sit on top)
        fig.add_trace(go.Scattermapbox(
            lat=gps["lat_dec"], lon=gps["lon_dec"],
            mode="lines",
            line=dict(width=4, color="#3b82f6"),
            name="Route",
            hoverinfo="skip",
        ))

        # 2) Coloured points along the route
        if colour_col == "zone":
            # Categorical colouring — one trace per zone for legend support
            zone_order = [z_ for z_ in ZONE_COLOURS if z_ in gps["zone"].unique()]
            for zname in zone_order:
                zsub = gps[gps["zone"] == zname]
                fig.add_trace(go.Scattermapbox(
                    lat=zsub["lat_dec"], lon=zsub["lon_dec"],
                    mode="markers",
                    marker=dict(size=9, color=ZONE_COLOURS[zname]),
                    name=zname,
                    hovertext=zsub["timestamp"].dt.strftime("%H:%M:%S") +
                              " — " + zsub["tempC"].round(1).astype(str) + " °C",
                    hoverinfo="text",
                ))
        else:
            # Continuous colouring
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

        # 3) Start marker (green)
        start = gps.iloc[0]
        fig.add_trace(go.Scattermapbox(
            lat=[start["lat_dec"]], lon=[start["lon_dec"]],
            mode="markers+text",
            marker=dict(size=18, color="#10b981"),
            text=["Start"], textposition="top right",
            textfont=dict(size=14, color="#10b981"),
            name="Start",
            hovertext=f"Start: {start['timestamp']:%H:%M:%S}",
            hoverinfo="text",
        ))

        # 4) End marker (red)
        end = gps.iloc[-1]
        fig.add_trace(go.Scattermapbox(
            lat=[end["lat_dec"]], lon=[end["lon_dec"]],
            mode="markers+text",
            marker=dict(size=18, color="#ef4444"),
            text=["End"], textposition="top right",
            textfont=dict(size=14, color="#ef4444"),
            name="End",
            hovertext=f"End: {end['timestamp']:%H:%M:%S}",
            hoverinfo="text",
        ))

        # Auto-zoom to fit the track
        center_lat = gps["lat_dec"].mean()
        center_lon = gps["lon_dec"].mean()
        # Rough zoom heuristic from bounding-box span
        span = max(
            gps["lat_dec"].max() - gps["lat_dec"].min(),
            gps["lon_dec"].max() - gps["lon_dec"].min(),
        )
        zoom = 15 if span < 0.01 else 14 if span < 0.03 else 13

        fig.update_layout(
            mapbox=dict(style=map_style, center=dict(lat=center_lat, lon=center_lon), zoom=zoom),
            height=600,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=(colour_col == "zone"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Speed-over-time chart ------------------------------------
        with st.expander("📊 Speed over time", expanded=False):
            speed_fig = px.line(
                gps, x="timestamp", y="speed_kmh",
                title=None,
                labels={"speed_kmh": "Speed (km/h)", "timestamp": "Time"},
            )
            speed_fig.update_traces(line=dict(color="#3b82f6", width=2))
            speed_fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(speed_fig, use_container_width=True)
            st.caption(
                f"Speeds are derived from haversine distance between consecutive "
                f"GPS fixes (~5s apart). Walking pace is typically 4–6 km/h; "
                f"anything above ~10 km/h is usually GPS jitter rather than real movement."
            )

# ---- Zone analysis -------------------------------------------------------
with tab_zones:
    st.subheader("Micro-climate comparison across zones")
    st.caption(
        "Each GPS fix is assigned to a zone (Museumplein, Frans Halsbuurt, Sarphatipark) "
        "or *Transit* if it falls outside all three bounding boxes. Edit the `ZONES` dict "
        "at the top of `app.py` to refine boundaries."
    )

    # Filter — drop transit if requested, drop moving samples if requested
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

    if z["zone"].nunique() < 1 or len(z) == 0:
        st.warning("No samples match the current filters.")
    else:
        # --- Summary table -------------------------------------------------
        summary = (z.groupby("zone")
                    .agg(samples=("tempC", "count"),
                         mean_temp=("tempC", "mean"),
                         std_temp=("tempC", "std"),
                         min_temp=("tempC", "min"),
                         max_temp=("tempC", "max"),
                         median_light=("light_lux", "median"),
                         max_light=("light_lux", "max"),
                         mean_mq=("mq_raw", "mean"))
                    .round(2))
        st.markdown("**Summary by zone**")
        st.dataframe(summary, use_container_width=True)

        # --- Boxplot of temperature ---------------------------------------
        st.markdown("**Temperature distribution by zone**")
        zone_order = [z_ for z_ in ZONE_COLOURS if z_ in z["zone"].unique()]
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

        # --- Light vs Temp scatter, coloured by zone ----------------------
        st.markdown("**Light explains the temperature difference**")
        fig_sc = px.scatter(
            z.dropna(subset=["light_lux", "tempC"]),
            x="light_lux", y="tempC", color="zone",
            category_orders={"zone": zone_order},
            color_discrete_map=ZONE_COLOURS,
            log_x=True, opacity=0.7,
            labels={"light_lux": "Light (lux, log scale)", "tempC": "Temperature (°C)"},
        )
        fig_sc.update_traces(marker=dict(size=8))
        fig_sc.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10),
                             legend_title_text="")
        st.plotly_chart(fig_sc, use_container_width=True)

        # --- Time-of-day exposure (the confounder) ------------------------
        st.markdown("**When was each zone visited?** *(time-of-day confounder check)*")
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
            "temperature difference may be time-of-day rather than location. Mention "
            "this caveat in your discussion section."
        )


# ---- Correlations --------------------------------------------------------
with tab_corr:
    st.subheader("How do sensors relate?")

    cols = ["tempC", "pressure_hPa", "light_lux", "mq_raw"]
    corr = df[cols].corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        aspect="auto",
    )
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Pairwise scatter**")
    c1, c2 = st.columns(2)
    x_axis = c1.selectbox("X axis", cols, index=0)
    y_axis = c2.selectbox("Y axis", cols, index=2)

    fig2 = px.scatter(
        df, x=x_axis, y=y_axis,
        trendline="ols",
        opacity=0.6,
        height=420,
    )
    fig2.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig2, use_container_width=True)

# ---- Raw data ------------------------------------------------------------
with tab_data:
    st.subheader("Filtered records")
    st.dataframe(df, use_container_width=True, height=500)

    csv_bytes = df.to_csv(index=False).encode()
    st.download_button(
        "⬇️ Download filtered CSV",
        data=csv_bytes,
        file_name="my_session.csv",
        mime="text/csv",
    )