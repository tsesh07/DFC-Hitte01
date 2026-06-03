"""
app.py — Navigatie-router voor het Future Cities hitte-dashboard.
================================================================
Dit bestand is bewust dun: het regelt alleen de paginaconfiguratie en de
sidebar-navigatie via st.navigation. De daadwerkelijke inhoud leeft in:

    home_dashboard.py      → 📡 Sensor-dashboard (hoofdpagina)
    views/Correlaties.py   → 🔗 Correlaties & scatterplots
    views/Interpolatie.py  → 🗺️ Geospatiale interpolatie

Elke pagina krijgt hieronder een eigen titel + icoon. st.set_page_config
mag maar één keer per run, dus dat staat alleen hier (niet in de pagina's).
"""
import streamlit as st

st.set_page_config(
    page_title="Future Cities — Hittedashboard",
    page_icon="🏙️",
    layout="wide",
)

# --------------------------------------------------------------------------
# Pagina's — titel + icoon bepalen wat er in de sidebar-navigatie verschijnt
# --------------------------------------------------------------------------
pagina_dashboard = st.Page(
    "home_dashboard.py",
    title="Sensor-dashboard",
    icon="📡",
    default=True,
)
pagina_correlaties = st.Page(
    "views/Correlaties.py",
    title="Correlaties & scatterplots",
    icon="🔗",
)
pagina_interpolatie = st.Page(
    "views/Interpolatie.py",
    title="Geospatiale interpolatie",
    icon="🗺️",
)

# Groeperen onder een kopje maakt de navigatie prominenter en duidelijker
navigatie = st.navigation(
    {"📂 Analyses": [pagina_dashboard, pagina_correlaties, pagina_interpolatie]}
)

navigatie.run()
