import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import geopandas as gpd
import seaborn as sns            # Voor de lichte statische plots
import matplotlib.pyplot as plt  # Nodig voor het donkere thema en sluiten van figuren
from pathlib import Path

# Pad-helpers: __file__ → views/Correlaties.py, .parents[1] = project-root.
# Zo werkt alles ongeacht de CWD waarvandaan Streamlit start.
APP_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH       = APP_ROOT / "data" / "processed" / "DATA_na_5e_meting_interpolatie.csv"
BUURTEN_GEOJSON = APP_ROOT / "data" / "zone"      / "gekozen_buurten_metbuffer.geojson"

# Pagina-config wordt centraal in app.py geregeld (st.navigation-router).
# De originele CSS gebruikte 'margin-top: -2rem', wat de inhoud onder de
# (verborgen) header omhoogtrok en bovenaan afkapte zodra de pagina in de
# multipage-router draait. Vervangen door positieve top-padding zodat de
# titel/tabbladen netjes vrijkomen.
st.markdown("""<style>[data-testid="stHeader"] {background: rgba(0,0,0,0);height: 0rem;position: fixed;z-index: 999;}[data-testid="stHeader"] [data-testid="stActionButton"] {display: none;}.block-container {padding-top: 3rem;padding-bottom: 0rem;}[data-testid="stSidebarUserContent"] {padding-top: 1rem;}</style>""", unsafe_allow_html=True)

# ==============================================================================
# 2. DATA INLADEN EN VOORBEWERKEN
# ==============================================================================
@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH)

    # Datetime converteren en datum extraheren voor de filter
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date

    # NMEA GPS omzetten naar Decimale Graden (DD.DDDDD)
    def nmea_to_decimal(v):
        if pd.isna(v) or v == 0:
            return None
        degrees = int(v / 100)
        minutes = v - (degrees * 100)
        return degrees + (minutes / 60)

    df['latitude'] = df['lat'].apply(nmea_to_decimal)
    df['longitude'] = df['lon'].apply(nmea_to_decimal)

    # GEOJSON INLADEN & PUNTEN KOPPELEN AAN BUURTEN
    try:
        gdf_buurten = gpd.read_file(BUURTEN_GEOJSON)

        if gdf_buurten.crs != "EPSG:4326":
            gdf_buurten = gdf_buurten.to_crs("EPSG:4326")

        df_gps = df.dropna(subset=['latitude', 'longitude']).copy()
        gdf_punten = gpd.GeoDataFrame(
            df_gps,
            geometry=gpd.points_from_xy(df_gps['longitude'], df_gps['latitude']),
            crs="EPSG:4326"
        )

        gdf_joined = gpd.sjoin(gdf_punten, gdf_buurten, how='left', predicate='within')

        # De gebufferde buurtgrenzen overlappen elkaar aan de randen, waardoor
        # één meetpunt aan meerdere buurten kan worden gekoppeld. sjoin geeft
        # dan twee rijen met dezelfde index — die zorgen later bij toewijzing
        # voor 'cannot reindex on an axis with duplicate labels'. We houden per
        # punt alleen de eerste match.
        gdf_joined = gdf_joined[~gdf_joined.index.duplicated(keep='first')]

        # Aangepast naar jouw nieuwe kolomnaam 'buurtname'
        buurt_kolomnaam_in_geojson = 'buurtnaam'

        if buurt_kolomnaam_in_geojson in gdf_joined.columns:
            df['buurt'] = gdf_joined[buurt_kolomnaam_in_geojson]
        else:
            beschikbare_kolommen = [c for c in gdf_buurten.columns if c != 'geometry']
            if beschikbare_kolommen:
                df['buurt'] = gdf_joined[beschikbare_kolommen[0]]
            else:
                df['buurt'] = 'Vlak zonder naam'

        df['buurt'] = df['buurt'].fillna('Buiten buurten')

    except Exception as geo_e:
        st.sidebar.error(f"Fout bij koppelen GeoJSON: {geo_e}")
        df['buurt'] = 'Geen buurtdata beschikbaar'

    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error(
        f"Het databestand voor deze pagina ontbreekt:\n\n`{CSV_PATH.relative_to(APP_ROOT)}`\n\n"
        "Plaats `DATA_na_5e_meting_interpolatie.csv` in `data/processed/` en herlaad."
    )
    st.stop()
except Exception as e:
    st.error(f"Kon het bestand niet vinden of openen. Fout: {e}")
    st.stop()

# ==============================================================================
# 3. SIDEBAR VOOR FILTERS EN INSTELLINGEN
# ==============================================================================
st.sidebar.header("⚙️ Instellingen")

# Dag selecteren
beschikbare_dagen = df['date'].unique()
gekozen_dag = st.sidebar.selectbox("Kies een dag:", beschikbare_dagen)

# LUX FILTER TOEVOEGEN
st.sidebar.markdown("---")
st.sidebar.subheader("🔦 Licht Filter")
lux_filter_waarde = st.sidebar.number_input(
    "Filter Lux boven:",
    value=250,
    help="Filtert alle waarden boven deze drempel weg."
)
st.sidebar.caption("deksel dicht: 250 & deksel open: 60000")

# Filter de dataset op de gekozen dag
df_dag = df[df['date'] == gekozen_dag].copy()
df_dag = df_dag.sort_values('timestamp').reset_index(drop=True)

# ==============================================================================
# TEMPERATUREN, LUX & PRESSURE BEREKENEN & FILTEREN
# ==============================================================================
df_dag['tempC_clean'] = df_dag['tempC'].copy()

# Filter onrealistische temperaturen
df_dag.loc[(df_dag['tempC_clean'] > 40) | (df_dag['tempC_clean'] < 5), 'tempC_clean'] = np.nan

# Bereken de mediaan op basis van de schone kolom
dag_mediaan = df_dag['tempC_clean'].median()

# Bereken het verschil ten opzichte van die mediaan
df_dag['temp_av_stdv'] = df_dag['tempC_clean'] - dag_mediaan

# Outlier filter voor temperatuur verschil
df_dag.loc[df_dag['temp_av_stdv'].abs() > 10, 'temp_av_stdv'] = np.nan

# Licht lux filteren op basis van de input
df_dag.loc[df_dag['light_lux'].abs() > lux_filter_waarde, 'light_lux'] = np.nan

# Jouw pressure filter rond 1000
df_dag.loc[(df_dag['pressure_hPa'] > 1100) | (df_dag['pressure_hPa'] < 900), 'pressure_hPa'] = np.nan

# ==============================================================================
# COLUMNS SELECTIE VOOR SCATTERPLOTS
# ==============================================================================
alle_numeriek = df_dag.select_dtypes(include=[np.number]).columns.tolist()
negeer_cols = ['tempC', 'tempC_clean', 'temp_av_stdv', 'lat', 'lon', 'latitude', 'longitude']
plot_opties = [c for c in alle_numeriek if c not in negeer_cols]

# ==============================================================================
# TABS AANMAKEN
# ==============================================================================
tab2, tab3, tab1 = st.tabs([
    "🔍 Interactief Scatter Analyse",
    "📊 Alle Correlaties (Seaborn)",
    "📈 Dashboard",
])

# ------------------------------------------------------------------------------
# TAB 1: DASHBOARD
# ------------------------------------------------------------------------------
with tab1:
    st.title("Temperatuur verschillen in de stad Amsterdam")

    df_kaart = df_dag.dropna(subset=['latitude', 'longitude']).copy()
    df_kaart = df_kaart[(df_kaart['latitude'] > 51.5) & (df_kaart['longitude'] > 4.0)]
    df_kaart = df_kaart.reset_index(drop=True)

    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c

    totale_afstand_km = 0.0
    tijdsduur_str = "Onbekend"

    if len(df_kaart) > 1:
        afstanden = haversine_distance(
            df_kaart['latitude'].shift(), df_kaart['longitude'].shift(),
            df_kaart['latitude'], df_kaart['longitude']
        )
        totale_afstand_km = afstanden.sum()
        start_tijd = df_kaart['timestamp'].iloc[0]
        eind_tijd = df_kaart['timestamp'].iloc[-1]
        duur = eind_tijd - start_tijd
        uren = duur.seconds // 3600
        minuten = (duur.seconds % 3600) // 60
        tijdsduur_str = f"{uren}u {minuten}m" if uren > 0 else f"{minuten}m"

    maanden_nl = {
        "January": "januari", "February": "februari", "March": "maart",
        "April": "april", "May": "mei", "June": "juni",
        "July": "juli", "August": "augustus", "September": "september",
        "October": "oktober", "November": "november", "December": "december"
    }
    maand_engels = gekozen_dag.strftime('%B')
    maand_nederlands = maanden_nl.get(maand_engels, maand_engels)
    datum_mooi = f"{gekozen_dag.strftime('%d')} {maand_nederlands}"

    st.subheader(f"📊 Statistieken van de wandeling op {gekozen_dag} ({datum_mooi})")
    kpi1, kpi2, kpi3, kpi3a, kpi4 = st.columns(5)

    with kpi1:
        st.metric(label="⏱️ Totale Tijdsduur", value=tijdsduur_str)
    with kpi2:
        if totale_afstand_km < 1.0:
            st.metric(label="📏 Afstand", value=f"{int(totale_afstand_km * 1000)} meter")
        else:
            st.metric(label="📏 Afstand", value=f"{totale_afstand_km:.2f} km")
    with kpi3:
        gem_temp = df_dag['tempC_clean'].mean()
        st.metric(label="🌡️ Gem. Temperatuur", value=f"{gem_temp:.1f} °C" if not pd.isna(gem_temp) else "N/B")
    with kpi3a:
        med_temp = df_dag['tempC_clean'].median()
        st.metric(label="🌡️ mediaan. Temperatuur", value=f"{med_temp:.1f} °C")
    with kpi4:
        st.metric(label="📍 Meetpunten", value=f"{len(df_kaart)} locaties")

    st.markdown("---")

    kleur_opties = {
        'Tijd': 'timestamp',
        'Temperatuur Afwijking': 'temp_av_stdv',
        'Temperatuur (schoon °C)': 'tempC_clean',
        'Luchtdruk (hPa)': 'pressure_hPa',
        'Lichtsterkte (Lux)': 'light_lux',
        'MQ Sensor (Raw)': 'mq_raw',
        'Gekoppelde Buurt': 'buurt'
    }
    gekozen_kleur = st.sidebar.selectbox("Kleur van de punten op de kaart gebaseerd op:", list(kleur_opties.keys()))
    kleur_kolom = kleur_opties[gekozen_kleur]

    if kleur_kolom == 'timestamp':
        df_kaart['kleur_waarde'] = df_kaart['timestamp'].astype('int64') // 10**9
        kleuren_schaal = 'Viridis'
        cmid_waarde = None
    elif kleur_kolom in ['temp_av_stdv', 'tempC_clean']:
        df_kaart['kleur_waarde'] = df_kaart[kleur_kolom]
        kleuren_schaal = 'RdBu_r'
        cmid_waarde = 0.0 if kleur_kolom == 'temp_av_stdv' else None
    elif kleur_kolom == 'buurt':
        df_kaart['kleur_waarde'] = df_kaart['buurt']
        kleuren_schaal = None
        cmid_waarde = None
    else:
        df_kaart['kleur_waarde'] = pd.to_numeric(df_kaart[kleur_kolom], errors='coerce')
        kleuren_schaal = 'Viridis'
        cmid_waarde = None

    kaart_stijlen = {
        "Donker (CartoDB Dark Matter)": "carto-darkmatter",
        "OpenStreetMap": "open-street-map",
        "Licht (CartoDB Positron)": "carto-positron",
        "Luchtfoto / Satelliet": "satellite-streets"
    }
    gekozen_stijl = st.sidebar.selectbox("Kies kaart achtergrond:", list(kaart_stijlen.keys()), index=0)
    stijl_id = kaart_stijlen[gekozen_stijl]

    st.subheader("🗺️ Route & Temperatuurkaart")
    if not df_kaart.empty:
        fig_map = go.Figure()
        shared_hover_template = (
            "<b>Tijd:</b> %{hovertext}<br>" +
            "<b>Buurt:</b> %{customdata[5]}<br>" +
            "<b>Echte Temp:</b> %{customdata[0]:.2f} °C<br>" +
            "<b>Schone Temp:</b> %{customdata[1]:.2f} °C<br>" +
            "<b>Verschil m.b.t. mediaan:</b> %{customdata[2]:.2f} °C<br>" +
            "<b>Druk:</b> %{customdata[3]} hPa<br>" +
            "<b>Licht:</b> %{customdata[4]} Lux<br>" +
            "<extra></extra>"
        )

        fig_map.add_trace(go.Scattermap(
            lat=df_kaart['latitude'], lon=df_kaart['longitude'], mode='lines',
            line=dict(color='rgba(255, 255, 255, 0.4)' if stijl_id == 'carto-darkmatter' else 'rgba(0, 0, 0, 0.3)', width=3),
            name='Gelopen Route', showlegend=True, hoverinfo='skip'
        ))

        if kleur_kolom == 'buurt':
            for buurt_naam in df_kaart['buurt'].unique():
                df_buurt_filter = df_kaart[df_kaart['buurt'] == buurt_naam]
                fig_map.add_trace(go.Scattermap(
                    lat=df_buurt_filter['latitude'], lon=df_buurt_filter['longitude'], mode='markers',
                    marker=dict(size=9),
                    hovertext=df_buurt_filter['timestamp'].dt.strftime('%H:%M:%S'),
                    customdata=df_buurt_filter[['tempC', 'tempC_clean', 'temp_av_stdv', 'pressure_hPa', 'light_lux', 'buurt']],
                    hovertemplate=shared_hover_template, name=buurt_naam
                ))
        else:
            fig_map.add_trace(go.Scattermap(
                lat=df_kaart['latitude'], lon=df_kaart['longitude'], mode='markers',
                marker=dict(size=8, color=df_kaart['kleur_waarde'], colorscale=kleuren_schaal, cmid=cmid_waarde, showscale=True),
                hovertext=df_kaart['timestamp'].dt.strftime('%H:%M:%S'),
                customdata=df_kaart[['tempC', 'tempC_clean', 'temp_av_stdv', 'pressure_hPa', 'light_lux', 'buurt']],
                hovertemplate=shared_hover_template, name='Meetpunten', showlegend=False
            ))

        fig_map.add_trace(go.Scattermap(
            lat=[df_kaart['latitude'].iloc[0]], lon=[df_kaart['longitude'].iloc[0]], mode='markers+text',
            marker=dict(color='#10B981', size=16), text=["Start"], textposition="top right", name='START', showlegend=False
        ))
        fig_map.add_trace(go.Scattermap(
            lat=[df_kaart['latitude'].iloc[-1]], lon=[df_kaart['longitude'].iloc[-1]], mode='markers+text',
            marker=dict(color='#EF4444', size=16), text=["Eind"], textposition="top right", name='EIND', showlegend=False
        ))

        fig_map.update_layout(height=700, map=dict(style=stijl_id, center=dict(lat=52.3567817, lon=4.8897621), zoom=14.9), margin={"r":50,"t":40,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.warning("Geen geldige GPS data gevonden.")

# ==============================================================================
    # BOXPLOT TOEVOEGEN (GECOMBINEERDE SPECIFIEKE DAGEN)
    # ==============================================================================
    st.markdown("---")
    st.subheader("📊 Temperatuurverdelingen per Buurt (Gecombineerde Dagen)")

    # Filteren op de 3 gewenste datums
    specifieke_dagen = [pd.to_datetime(d).date() for d in ['2026-05-06', '2026-05-08', '2026-05-20']]
    df_boxplot = df[df['date'].isin(specifieke_dagen)].copy()

    if not df_boxplot.empty:
        # 1. Opschonen van de basis temperatuur
        df_boxplot['tempC_clean'] = df_boxplot['tempC'].copy()
        df_boxplot.loc[(df_boxplot['tempC_clean'] > 40) | (df_boxplot['tempC_clean'] < 5), 'tempC_clean'] = np.nan

        # 2. Bereken de mediaan van deze 3 dagen samen voor de afwijking
        boxplot_mediaan = df_boxplot['tempC_clean'].median()
        df_boxplot['temp_av_stdv'] = df_boxplot['tempC_clean'] - boxplot_mediaan

        # Outlier filter voor temperatuur verschil (consistent met de rest van de app)
        df_boxplot.loc[df_boxplot['temp_av_stdv'].abs() > 10, 'temp_av_stdv'] = np.nan

        # Drop NaN waarden voor de plot
        df_boxplot = df_boxplot.dropna(subset=['tempC_clean', 'temp_av_stdv', 'buurt'])

        # --- BOXPLOT 1: Schone Temperatuur (°C) — alleen de geselecteerde dag ---
        fig_box = px.box(
            df_dag,
            x='buurt',
            y='tempC_clean',
            points="all",
            title=f"Temperatuur spreiding per buurt voor {gekozen_dag}",
            labels={'buurt': 'Buurten', 'tempC_clean': 'Temperatuur (°C)'},
            color='buurt',
            color_discrete_sequence=px.colors.qualitative.Bold,
            hover_data=['lon', 'lat']
        )
        fig_box.update_layout(height=600, xaxis_title="Buurt", yaxis_title="Temperatuur (°C)", showlegend=False)
        fig_box.update_yaxes(tickfont=dict(size=20), title_font=dict(size=20))
        st.plotly_chart(fig_box, use_container_width=True)

        # --- BOXPLOT 2: Temperatuur Afwijking ---
        fig_box_stdv = px.box(
            df_boxplot,
            x='buurt',
            y='temp_av_stdv',
            points="all",
            title="Temperatuurafwijking t.o.v. mediaan per buurt voor 6, 8 en 20 mei 2026 (Samen)",
            labels={'buurt': 'Buurten', 'temp_av_stdv': 'Temperatuur Afwijking (°C)'},
            color='buurt',
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        # Bij een afwijking is een nul-lijn (het nulpunt van de mediaan) visueel erg sterk:
        fig_box_stdv.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Mediaan (0 °C)")

        fig_box_stdv.update_layout(height=600, xaxis_title="Buurt", yaxis_title="Temperatuur Afwijking (°C)", showlegend=False)
        fig_box_stdv.update_yaxes(tickfont=dict(size=20), title_font=dict(size=20))
        st.plotly_chart(fig_box_stdv, use_container_width=True)

    else:
        st.info("Geen data beschikbaar voor de datums: 2026-05-06, 2026-05-08, 2026-05-20.")

    # ==============================================================================
    # OUDE GRAFIEKJES ONDER DE BOXPLOT
    # ==============================================================================
    st.markdown("---")
    st.subheader("📊 Sensormetingen over de tijd")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.line(df_dag, x='timestamp', y='tempC_clean', title="Temperatuur verloop (schoon)", color_discrete_sequence=['#ef553b']), use_container_width=True)
        st.plotly_chart(px.line(df_dag, x='timestamp', y='light_lux', title="Lichtsterkte (Lux)", color_discrete_sequence=['#636efa']), use_container_width=True)
    with col2:
        st.plotly_chart(px.line(df_dag, x='timestamp', y='pressure_hPa', title="Luchtdruk verloop", color_discrete_sequence=['#00cc96']), use_container_width=True)
        st.plotly_chart(px.line(df_dag, x='timestamp', y='mq_raw', title="Gas Sensor (MQ Raw)", color_discrete_sequence=['#ab63fa']), use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 2: INTERACTIEF SCATTER ANALYSE
# ------------------------------------------------------------------------------
with tab2:
    st.header("🔍 Interactief Scatter Analyse")
    st.markdown("Kies zelf de assen en kleurstellingen om patronen interactief te ontdekken.")

    if plot_opties:
        col_x, col_color = st.columns(2)

        with col_x:
            geselecteerde_x = st.selectbox("Selecteer variabele voor de X-as:", plot_opties, index=0)

        with col_color:
            scatter_kleur_opties = {
                'Gekoppelde Buurt': 'buurt',
                'Temperatuur (schoon °C)': 'tempC_clean',
                'Temperatuur Afwijking': 'temp_av_stdv',
                'Luchtdruk (hPa)': 'pressure_hPa',
                'Lichtsterkte (Lux)': 'light_lux',
                'MQ Sensor (Raw)': 'mq_raw'
            }
            gekozen_scatter_kleur = st.selectbox(
                "Kleur van de punten in de scatterplot gebaseerd op:",
                list(scatter_kleur_opties.keys()),
                index=0,
                key="interactieve_kleur_kiezer"
            )

        scatter_kleur_kolom = scatter_kleur_opties[gekozen_scatter_kleur]

        scatter_kwargs = {
            'data_frame': df_dag,
            'x': geselecteerde_x,
            'y': 'tempC_clean',
            'color': scatter_kleur_kolom,
            'title': f"Scatterplot: {geselecteerde_x} vs Temperatuur (Gekleurd op: {gekozen_scatter_kleur})",
            'labels': {
                'tempC_clean': 'Temperatuur (°C)',
                geselecteerde_x: geselecteerde_x,
                scatter_kleur_kolom: gekozen_scatter_kleur
            },
            'hover_data': ['timestamp']
        }

        if scatter_kleur_kolom == 'buurt':
            scatter_kwargs['color_discrete_sequence'] = px.colors.qualitative.Bold
        else:
            scatter_kwargs['color_continuous_scale'] = 'RdBu_r'

        fig_scatter = px.scatter(**scatter_kwargs)
        fig_scatter.update_layout(height=600)
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.warning("Niet genoeg numerieke data gevonden.")

# ------------------------------------------------------------------------------
# TAB 3: ALLE COMBINATIES (STATISCH VIA SEABORN MET DONKER THEMA)
# ------------------------------------------------------------------------------
with tab3:
    st.header("📊 Statische Matrix Scatter Analyse (Seaborn)")
    st.markdown(
        "Hieronder zie je de **volledige 4x4 matrix (16 variaties)**. "
        "Elke sensorwaarde op de X-as wordt gecombineerd met alle 4 mogelijke kleur-indicatoren."
    )

    plt.style.use('dark_background')
    kleur_opties_seaborn = ['buurt', 'light_lux', 'pressure_hPa', 'mq_raw']

    if plot_opties:
        for x_col in plot_opties:
            st.subheader(f"🔄 Analysereeks voor X-as: `{x_col}`")
            st_cols = st.columns(4)

            for i, color_col in enumerate(kleur_opties_seaborn):
                fig, ax = plt.subplots(figsize=(6, 5))
                fig.patch.set_facecolor('#121212')
                ax.set_facecolor('#1e1e1e')

                if color_col == 'buurt':
                    current_palette = 'prism'
                else:
                    current_palette = 'magma'

                sns.scatterplot(
                    data=df_dag,
                    x=x_col,
                    y='tempC_clean',
                    hue=color_col,
                    palette=current_palette,
                    alpha=0.8,
                    edgecolor='none',
                    s=40,
                    ax=ax
                )

                ax.set_title(f"Kleur: {color_col}", fontsize=11, weight='bold', color='#ffffff', pad=8)
                ax.set_xlabel(x_col, fontsize=9, color="#ffffff")
                ax.set_ylabel("Temperatuur (°C)", fontsize=9, color="#ffffff")
                ax.tick_params(colors="#ffffff", labelsize=8)
                ax.grid(True, linestyle=':', alpha=0.3, color='#555555')

                leg = ax.legend(loc='upper right', framealpha=0.8, prop={'size': 7})
                if leg:
                    leg.get_frame().set_facecolor("#000000")
                    for text in leg.get_texts():
                        text.set_color('#ffffff')

                with st_cols[i]:
                    st.pyplot(fig, use_container_width=True, transparent=True)

                plt.close(fig)
    else:
        st.warning("Geen geschikte numerieke kolommen gevonden voor de automatische matrix loop.")
