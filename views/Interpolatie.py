import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.fill import fillnodata
import plotly.express as px
from pathlib import Path

# Pad-helper: __file__ → views/Interpolatie.py, .parents[1] = project-root.
APP_ROOT = Path(__file__).resolve().parents[1]
GEOJSON_PATH = APP_ROOT / "data" / "processed" / "DATA_gefilterd_amersfoort.geojson"

# Pagina-config wordt centraal in app.py geregeld (st.navigation-router).

st.title("🗺️ Interpolatie")
st.markdown("Kies een datum en test-split om de interpolatie te berekenen en de nauwkeurigheid te valideren.")

# Data inladen (met cache zodat het dashboard snel blijft)

@st.cache_data
def load_data(path):
    gdf = gpd.read_file(path)

    # Zorg dat het CRS expliciet op EPSG:28992 staat
    if gdf.crs is None:
        gdf.set_crs("EPSG:28992", inplace=True)
    elif gdf.crs.to_string() != "EPSG:28992":
        gdf = gdf.to_crs("EPSG:28992")

    # Zet 'timestamp' om naar een zuivere datum (YYYY-MM-DD)
    if 'timestamp' in gdf.columns:
        gdf['timestamp_datum'] = pd.to_datetime(gdf['timestamp']).dt.date

    return gdf

try:
    data_gefilterd_amersfoort = load_data(GEOJSON_PATH)
except FileNotFoundError:
    st.error(
        f"Het databestand voor deze pagina ontbreekt:\n\n"
        f"`{GEOJSON_PATH.relative_to(APP_ROOT)}`\n\n"
        "Plaats `DATA_gefilterd_amersfoort.geojson` in `data/processed/` en herlaad."
    )
    st.stop()
except Exception as e:
    st.error(f"Fout bij het laden van het GeoJSON-bestand: {e}")
    st.stop()


# 2. Sidebar voor de instellingen en filters
st.sidebar.header("⚙️ Instellingen")

# Datum selecteren
if 'timestamp_datum' in data_gefilterd_amersfoort.columns:
    date_col = 'timestamp_datum'
else:
    date_cols = [col for col in data_gefilterd_amersfoort.columns if 'dat' in col.lower() or 'date' in col.lower() or 'time' in col.lower()]
    date_col = st.sidebar.selectbox("Selecteer datumkolom:", date_cols if date_cols else data_gefilterd_amersfoort.columns)

unique_dates = sorted(data_gefilterd_amersfoort[date_col].dropna().unique())
selected_date = st.sidebar.selectbox("Kies een datum:", unique_dates)

# Filter de data op de gekozen datum
gdf_filtered = data_gefilterd_amersfoort[data_gefilterd_amersfoort[date_col] == selected_date]

st.sidebar.markdown("---")
st.sidebar.subheader("🔬 Validatie Instellingen")

# 🔥 NIEUW: Train/Test Split Slider
test_percent = st.sidebar.slider("Testset grootte (%)", min_value=5, max_value=50, value=10, step=5)

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Raster Parameters")
res = st.sidebar.slider("Grid Resolutie (meters)", min_value=1, max_value=5, value=1, step=1)
max_dist = st.sidebar.number_input("NoData Fill Radius (pixels)", min_value=1, max_value=500, value=100)
smooth_iters = st.sidebar.number_input("Smoothing Iterations", min_value=0, max_value=500, value=100)


# 3. De Hoofd-logica (Splitsing, Rasterisatie & Validatie)
if gdf_filtered.empty:
    st.warning(f"Geen datapunten gevonden voor {selected_date}.")
elif len(gdf_filtered) < 10:
    st.warning(f"Te weinig datapunten ({len(gdf_filtered)}) op deze dag om een betrouwbare train/test split te maken.")
else:
    st.sidebar.success(f"Totaal {len(gdf_filtered)} punten beschikbaar.")

    if st.sidebar.button("🚀 Start Analyse & Validatie"):
        with st.spinner("Data splitsen, interpoleren en valideren..."):

            # 🔥 A. Train / Test Split uitvoeren via Pandas sampling
            test_gdf = gdf_filtered.sample(frac=test_percent / 100.0, random_state=42)
            train_gdf = gdf_filtered.drop(test_gdf.index)

            st.write(f"📊 **Model setup:** {len(train_gdf)} punten gebruikt voor training, {len(test_gdf)} punten apart gehouden voor validatie.")

            # Grenzen bepalen op basis van de TOTALE gefilterde set (zodat testpunten binnen het grid vallen)
            minx, miny, maxx, maxy = gdf_filtered.total_bounds

            # Matrix dimensies berekenen
            width = int(np.ceil((maxx - minx) / res))
            height = int(np.ceil((maxy - miny) / res))

            if width * height > 10000000:
                st.error(f"Het gebied is te groot ({width}x{height} pixels). Verhoog de resolutie in de sidebar.")
                st.stop()

            # Transformatiematrix opzetten (Rijksdriehoekcoördinaten)
            transform = rasterio.transform.from_origin(minx, maxy, res, res)

            # 🔥 B. Rasteriseren met ALLEEN de TRAIN data
            train_shapes = [(geom, val) for geom, val in zip(train_gdf.geometry, train_gdf['verschil_met_mediaan'])]

            raster = rasterize(
                shapes=train_shapes,
                out_shape=(height, width),
                transform=transform,
                fill=np.nan,
                all_touched=True,
                dtype=np.float32
            )

            # Masker maken voor fillnodata
            mask = np.ones((height, width), dtype=np.uint8)
            mask[np.isnan(raster)] = 0

            # C. NoData gaten opvullen (Interpolatie van de train data)
            filled_raster = fillnodata(
                image=raster,
                mask=mask,
                max_search_distance=float(max_dist),
                smoothing_iterations=int(smooth_iters)
            )

            # 🔥 D. Model Validatie: Waarden opvragen voor de TEST-set
            test_actuals = []
            test_predictions = []

            for geom, actual_val in zip(test_gdf.geometry, test_gdf['verschil_met_mediaan']):
                # Haal de X en Y op in meters (Rijksdriehoek)
                x, y = geom.x, geom.y
                # Converteer de coördinaat naar de rij en kolom in de matrix
                row, col = rasterio.transform.rowcol(transform, x, y)

                # Check of het testpunt binnen de matrix valt
                if 0 <= row < height and 0 <= col < width:
                    pred_val = filled_raster[row, col]
                    if not np.isnan(pred_val):
                        test_actuals.append(actual_val)
                        test_predictions.append(pred_val)

            # Foutstatistieken berekenen
            if test_predictions:
                actuals = np.array(test_actuals)
                preds = np.array(test_predictions)

                mae = np.mean(np.abs(actuals - preds))
                rmse = np.sqrt(np.mean((actuals - preds) ** 2))

                # Toon metrics in mooie Streamlit kolommen
                col1, col2 = st.columns(2)
                col1.metric(label="📈 Mean Absolute Error (MAE)", value=f"{mae:.4f}", help="Gemiddelde absolute afwijking van je testpunten.")
                col2.metric(label="🎯 Root Mean Squared Error (RMSE)", value=f"{rmse:.4f}", help="Straft grotere fouten harder af.")
            else:
                st.warning("De testpunten konden niet gekoppeld worden aan het geïnterpoleerde raster.")

            # E. Matrix omzetten naar DataFrame voor visualisatie
            rows, cols = np.where(~np.isnan(filled_raster))
            xs, ys = rasterio.transform.xy(transform, rows, cols)

            df_raster = pd.DataFrame({
                'x': xs,
                'y': ys,
                'verschil_met_mediaan': filled_raster[rows, cols]
            })

            gdf_raster = gpd.GeoDataFrame(
                df_raster,
                geometry=gpd.points_from_xy(df_raster.x, df_raster.y),
                crs="EPSG:28992"
            )
            gdf_raster_4326 = gdf_raster.to_crs("EPSG:4326")

            gdf_raster_4326['lon'] = gdf_raster_4326.geometry.x
            gdf_raster_4326['lat'] = gdf_raster_4326.geometry.y

            # Downsampling voor Plotly performance
            if len(gdf_raster_4326) > 50000:
                st.info("De kaart bevat erg veel cellen. We downsamplen de weergave tot 50.000 punten voor een soepele ervaring.")
                gdf_raster_4326 = gdf_raster_4326.sample(n=50000, random_state=42)

            # 🔥 F. De NIEUWE Plotly Express scatter_map renderen
            fig = px.scatter_map(
                gdf_raster_4326,
                lat="lat",
                lon="lon",
                color="verschil_met_mediaan",
                color_continuous_scale="RdBu_r",
                zoom=13,
                title=f"Geinterpoleerd grid (getraind op {100-test_percent}%) voor datum: {selected_date}"
            )

            fig.update_layout(
                map_style="open-street-map",
                margin={"r": 0, "t": 40, "l": 0, "b": 0}
            )

            st.plotly_chart(fig, use_container_width=True)

# Tabel tonen onderaan ter controle
with st.expander("Toon brondata overzicht van deze dag"):
    st.dataframe(gdf_filtered)
