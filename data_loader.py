"""
data_loader.py — Eén bron van waarheid voor het inladen van de Arduino sensor-data.

Leest ../data/raw/DATA.CSV (de export van de SD-kaart, nooit aangepast) en
levert een schoongemaakte DataFrame voor het dashboard. Alle transformaties
leven hier zodat de pipeline reproduceerbaar is vanaf het originele bestand.
"""
from pathlib import Path
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).parent
RAW_PATH = APP_DIR / "data" / "raw" / "DATA.CSV"

# Elke wandeling identificeren we via de datum in de timestamp-kolom
SESSIONS = {
    "2026-05-06": "Dag 1 - deksel dicht",
    "2026-05-08": "Dag 2 - deksel open",
    "2026-05-18": "Dag 3 - GPS track",
    "2026-05-20": "Dag 4 - deksel dicht",
}

# Sessies waarvoor alleen de GPS-route beschikbaar is (sensormeting defect).
# Worden getoond op de kaart maar uitgesloten van alle sensor-analyses.
GPS_ONLY_SESSIONS = frozenset({"Dag 3 - GPS track"})

# Methodologische context per sessie — leeft hier zodat het dashboard en
# eventuele analyse-scripts dezelfde bron raadplegen. Vul bij elke nieuwe
# run dit dict aan met de relevante experimentele omstandigheden.
SESSION_METADATA = {
    "Dag 1 - deksel dicht": {
        "datum":           "2026-05-06",
        "wandel_richting": "Sarphatipark → Frans Halsbuurt → Museumplein",
        "deksel":          "Dicht (UV-/IR-attenuatie)",
        "hardware_status": "Stabiel — continue logging, geen onderbrekingen",
        "bekende_issues":  "Sensor-zelfopwarming: ca. +2.2 °C drift over wandeling",
        "rol_in_analyse":  "Primaire schone dataset",
    },
    "Dag 2 - deksel open": {
        "datum":           "2026-05-08",
        "wandel_richting": "Museumplein → Frans Halsbuurt → Sarphatipark "
                           "(omgekeerd t.o.v. Dag 1)",
        "deksel":          "Open",
        "hardware_status": "USB-C stroomstoringen (beschadigde kabel)",
        "bekende_issues":  "RTC-glitch timestamps hersteld (33 rijen); "
                           "temperatuurpieken bij power-dropouts gemaskeerd (>50 °C)",
        "rol_in_analyse":  "Route-omkering dient als controle voor sensor-drift; "
                           "data-kwaliteit lager dus secundair",
    },
    "Dag 3 - GPS track": {
        "datum":           "2026-05-18",
        "wandel_richting": "Onbekend — route niet volledig gedocumenteerd",
        "deksel":          "—",
        "hardware_status": "BMP280 I2C-bus uitval na ~22 min (losse kabel losgetrokken "
                           "door teamlid); Xtorm 35 Wh powerbank + nieuwe USB-C kabel",
        "bekende_issues":  "Sensormeting onbruikbaar na 15:32 UTC+2 (bevroren register, "
                           "herhalende waarden 189.10 °C / 1774.47 hPa); alleen GPS intact",
        "rol_in_analyse":  "GPS-only — uitgesloten van sensor-analyse; "
                           "route zichtbaar op kaart als stippellijn",
    },
    "Dag 4 - deksel dicht": {
        "datum":           "2026-05-20",
        "wandel_richting": "Sarphatipark → Frans Halsbuurt → Museumplein "
                           "(zelfde richting als Dag 1)",
        "deksel":          "Dicht (UV-/IR-attenuatie)",
        "hardware_status": "Stabiel — Xtorm 35 Wh powerbank + USB-C, continue logging "
                           "zonder onderbrekingen",
        "bekende_issues":  "Geen — schoonste dataset tot nu toe (0 temperatuur-uitschieters, "
                           "geen drukdropouts); ~3.5 min stilstaande warmup aan het begin weggeknipt",
        "rol_in_analyse":  "Directe replicatie van Dag 1 (zelfde route + deksel dicht): "
                           "toetst of het ruimtelijke patroon reproduceerbaar is",
    },
}

# Wandel-vensters (gps_time als HHMMSS UTC) — knipt opwarming/idle weg.
# Zet een grens op None om alle metingen voor die sessie te behouden.
#
# Dag 2: het apparaat stond al om 11:00 lokaal aan, maar de wandeling begon
# pas rond 13:00 lokaal (= 11:00 UTC = gps_time 110000). Voor die tijd is er
# alleen sensor-warmup zonder GPS-fix, dat willen we niet meenemen.
WALK_WINDOWS = {
    "Dag 1 - deksel dicht": (121155, 130545),
    "Dag 2 - deksel open":  (110000, 122635),
    "Dag 3 - GPS track":    (130750, 141602),
    "Dag 4 - deksel dicht": (115010, 133627),
}


def _nmea_to_decimal(coord, hemi):
    """NMEA DDDMM.MMMM + halfrond-letter -> decimale graden."""
    if pd.isna(coord) or pd.isna(hemi):
        return None
    deg = int(coord // 100)
    minutes = coord - deg * 100
    dec = deg + minutes / 60
    return -dec if hemi in ("S", "W") else dec


def _repair_corrupt_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Herstelt RTC-glitch timestamps ('2001-51-14 ...') op basis van gps_time.

    Deze rijen vallen binnen de Dag-2-wandeling, dus reconstrueren we ze als
    2026-05-08 + HH:MM:SS uit gps_time. De gps_time-kolom komt rechtstreeks
    van de GPS-module en is niet aangetast door de RTC-storing.

    BELANGRIJK: gps_time staat in UTC, terwijl alle overige timestamps in
    lokale tijd (CEST = UTC+2) staan. We tellen daarom 2 uur op, anders landen
    de gerepareerde rijen 2 uur te vroeg (bv. 11:17 i.p.v. 13:17), wat de
    sessieduur en de tijdreeks vervuilt.
    """
    bad = df["timestamp"].astype(str).str.startswith("2001")
    if not bad.any():
        return df

    def _build(g):
        if pd.isna(g):
            return None
        g = int(g)
        h, m, s = g // 10000, (g // 100) % 100, g % 100
        # UTC -> lokale tijd (CEST); Timedelta vangt een eventuele uur-overloop op.
        t = pd.Timestamp("2026-05-08") + pd.Timedelta(hours=h + 2, minutes=m, seconds=s)
        return t.strftime("%Y-%m-%d %H:%M:%S")

    df.loc[bad, "timestamp"] = df.loc[bad, "gps_time"].apply(_build)
    return df


@st.cache_data
def load_data(path: Path = RAW_PATH) -> pd.DataFrame:
    """Laad en maak de ruwe SD-kaart export schoon.

    Pipeline:
      1. Lees CSV
      2. Verwijder volledig lege rijen (artefact van Excel-export)
      3. Herstel RTC-glitch timestamps uit gps_time
      4. Parse timestamp -> datetime; gooi onparseerbare rijen weg
      5. Filter naar de twee wandeldagen (verwijdert data van andere studenten)
      6. Voeg sessie-label toe
      7. Knip elke sessie naar het echte wandelvenster via gps_time
      8. Zet NMEA lat/lon om naar decimale graden
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