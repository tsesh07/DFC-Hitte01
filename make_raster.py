"""
make_raster.py — Geospatiale interpolatie van temperatuurmetingen → GeoTIFF
============================================================================
Methode : Inverse Distance Weighting (IDW, power = 2)
CRS     : RD New / EPSG:28992 (meters) — standaard voor Nederlandse geodata
Output  : output/<sessie>_temperature_idw.tif   (één bestand per sessie)

Gebruik:
    python3 make_raster.py

Vereisten (naast project-requirements):
    pip install rasterio
"""

from pathlib import Path
import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from pyproj import Transformer
import pyproj

# Zorg dat rasterio dezelfde PROJ-database gebruikt als pyproj.
# Dit voorkomt conflicten op systemen met meerdere PROJ-installaties
# (bv. Anaconda naast een systeeminstallatie).
os.environ.setdefault("PROJ_DATA", pyproj.datadir.get_data_dir())
os.environ.setdefault("PROJ_LIB",  pyproj.datadir.get_data_dir())

import rasterio
from rasterio.transform import from_origin

from data_loader import load_data, GPS_ONLY_SESSIONS

# Haal de CRS-WKT op via pyproj zodat we niet afhankelijk zijn van rasterio's
# ingebouwde PROJ-database (vermijdt conflicten met meerdere PROJ-installaties)
from pyproj import CRS as _PyprojCRS
_RD_NEW_WKT = _PyprojCRS.from_epsg(28992).to_wkt()

# ---------------------------------------------------------------------------
# Parameters — pas hier aan om resolutie / zoekstraal te wijzigen
# ---------------------------------------------------------------------------
RESOLUTION_M  = 20    # rastercelbreedte in meters
SEARCH_RADIUS = 200   # max. afstand (m) tot een meetpunt om mee te tellen in IDW
IDW_POWER     = 2     # standaard kwadratisch afstandsgewicht
BUFFER_M      = 100   # extra rand rondom de dataextent
OUTPUT_DIR    = Path(__file__).parent / "output"

# ---------------------------------------------------------------------------
# CRS-transformer: WGS84 (GPS) → RD New (meters)
# ---------------------------------------------------------------------------
_wgs84_to_rd = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)


def _lat_lon_to_rd(lat: np.ndarray, lon: np.ndarray):
    """Converteer WGS84 lat/lon-arrays naar RD New x/y (meters)."""
    x, y = _wgs84_to_rd.transform(lon, lat)
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def _drift_correct(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lineaire drift-correctie per sessie.

    De BMP280 warmt op tijdens de wandeling waardoor de temperatuur
    systematisch stijgt. Een lineaire fit per sessie verwijdert deze trend
    zodat het ruimtelijke signaal overblijft.
    """
    out = []
    for _, sub in df.groupby("session", sort=False):
        sub = sub.copy().sort_values("timestamp")
        t0    = sub["timestamp"].iloc[0]
        t_min = (sub["timestamp"] - t0).dt.total_seconds() / 60
        valid = sub["tempC"].notna() & t_min.notna()
        if valid.sum() > 2:
            slope, _ = np.polyfit(t_min[valid], sub.loc[valid, "tempC"], 1)
        else:
            slope = 0.0
        sub["tempC_detrended"] = sub["tempC"] - slope * t_min
        out.append(sub)
    return pd.concat(out).reset_index(drop=True)


def _idw_interpolate(x_pts: np.ndarray, y_pts: np.ndarray, z_pts: np.ndarray,
                     grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
    """
    Inverse Distance Weighting interpolatie.

    Voor elke rastercell worden de k dichtstbijzijnde meetpunten gezocht
    (max. SEARCH_RADIUS meter). Cellen buiten de zoekstraal van elk punt
    krijgen NaN — eerlijk over de gebieden waar geen data is.

    Parameters
    ----------
    x_pts, y_pts  : RD New coördinaten van de meetpunten
    z_pts         : gemeten waarden (drift-gecorrigeerde temperatuur)
    grid_x, grid_y: 2-D meshgrid van celcentra (RD New)

    Returns
    -------
    2-D array met geïnterpoleerde waarden (float32), NaN buiten bereik.
    """
    tree = cKDTree(np.column_stack([x_pts, y_pts]))
    flat = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    k = min(len(z_pts), 20)
    dists, idxs = tree.query(flat, k=k)

    # Cellen waarbij het dichtstbijzijnde punt verder is dan SEARCH_RADIUS → NaN
    outside = dists[:, 0] > SEARCH_RADIUS

    # Vermijd deling door nul voor exacte treffers
    dists = np.where(dists < 1e-6, 1e-6, dists)

    weights     = 1.0 / dists ** IDW_POWER
    weights    /= weights.sum(axis=1, keepdims=True)
    z_interp    = (weights * z_pts[idxs]).sum(axis=1).astype(np.float32)
    z_interp[outside] = np.nan

    return z_interp.reshape(grid_x.shape)


def make_session_raster(session_label: str, pts: pd.DataFrame,
                        output_path: Path) -> None:
    """
    Schrijf één GeoTIFF-raster voor een sessie.

    Parameters
    ----------
    session_label : naam van de sessie (voor metadata en logging)
    pts           : DataFrame met geldige lat_dec, lon_dec, tempC_detrended
    output_path   : uitvoerpad (.tif)
    """
    valid = pts.dropna(subset=["lat_dec", "lon_dec", "tempC_detrended"])
    if len(valid) < 4:
        print(f"  [SKIP] {session_label}: te weinig punten ({len(valid)})")
        return

    x_pts, y_pts = _lat_lon_to_rd(valid["lat_dec"].values, valid["lon_dec"].values)
    z_pts        = valid["tempC_detrended"].values

    # Rasterextent: meetpunten + buffer
    west  = x_pts.min() - BUFFER_M
    east  = x_pts.max() + BUFFER_M
    south = y_pts.min() - BUFFER_M
    north = y_pts.max() + BUFFER_M

    n_cols = int(np.ceil((east  - west)  / RESOLUTION_M))
    n_rows = int(np.ceil((north - south) / RESOLUTION_M))

    # Celcentra (rij 0 = noordrand)
    xs = west  + (np.arange(n_cols) + 0.5) * RESOLUTION_M
    ys = north - (np.arange(n_rows) + 0.5) * RESOLUTION_M
    grid_x, grid_y = np.meshgrid(xs, ys)

    print(f"  Rooster: {n_cols} × {n_rows} cellen  |  {len(valid)} meetpunten")
    raster = _idw_interpolate(x_pts, y_pts, z_pts, grid_x, grid_y)

    # Affiene transformatie: linkerbovenhoek van cel (0,0) = (west, north)
    transform = from_origin(west, north, RESOLUTION_M, RESOLUTION_M)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path, "w",
        driver="GTiff",
        height=n_rows,
        width=n_cols,
        count=1,
        dtype="float32",
        crs=_RD_NEW_WKT,
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(raster, 1)
        dst.update_tags(
            session=session_label,
            variable="tempC_detrended",
            units="degC (drift-gecorrigeerd)",
            methode=f"IDW power={IDW_POWER}, zoekstraal={SEARCH_RADIUS}m",
            resolutie=f"{RESOLUTION_M}m",
            crs="EPSG:28992 (RD New / Amersfoort)",
        )

    # Schrijf ook een .prj sidecar-bestand zodat GIS-software (QGIS, ArcGIS)
    # de projectie altijd herkent, ongeacht de PROJ-versie van de lezer.
    output_path.with_suffix(".prj").write_text(_RD_NEW_WKT, encoding="utf-8")

    print(f"  ✓  Opgeslagen: {output_path}")


# ---------------------------------------------------------------------------
# Hoofdprogramma
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("── Geospatiale rasterinterpolatie ──────────────────────────────")
    print(f"   Resolutie : {RESOLUTION_M} m")
    print(f"   Methode   : IDW (power={IDW_POWER}, radius={SEARCH_RADIUS} m)")
    print(f"   Variabele : tempC_detrended (drift-gecorrigeerde temperatuur)")
    print()

    print("Laden van sensordata…")
    raw = load_data()

    # Verwijder GPS-only sessies (geen bruikbare temperatuurdata)
    raw = raw[~raw["session"].isin(GPS_ONLY_SESSIONS)].copy()

    # Zelfde kwaliteitsfilters als het dashboard
    raw.loc[raw["tempC"] > 50,  "tempC"] = np.nan
    raw.loc[raw["tempC"] < -10, "tempC"] = np.nan
    raw.loc[raw["pressure_hPa"] < 800, "pressure_hPa"] = np.nan

    raw = _drift_correct(raw)

    print(f"Sessies gevonden: {list(raw['session'].unique())}\n")

    for session in raw["session"].unique():
        sub       = raw[raw["session"] == session]
        safe_name = session.lower().replace(" ", "_").replace("-", "").replace("(", "").replace(")", "")
        out_path  = OUTPUT_DIR / f"{safe_name}_temperature_idw.tif"
        print(f"→ {session}")
        make_session_raster(session, sub, out_path)
        print()

    print("── Klaar ───────────────────────────────────────────────────────")
    print(f"Bestanden staan in: {OUTPUT_DIR.resolve()}")
