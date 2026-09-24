import base64
import hashlib
import io
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image
try:
    from pyproj import CRS, Transformer
    HAS_PYPROJ = True
except Exception:
    CRS = None
    Transformer = None
    HAS_PYPROJ = False
import streamlit as st
import streamlit.components.v1 as components

# =========================================================
# STARTUP SETTINGS
# =========================================================
# Change this path to the folder containing the GeoTIFF/GeoJSON files you want
# the app to load automatically every time it starts.
DATA_FOLDER = Path(__file__).resolve().parent / "data"
AUTO_LOAD_FOLDER = True
SUPPORTED_DATA_EXTENSIONS = {".tif", ".tiff", ".geojson", ".json"}

# =========================================================
# RESILIENT RASTERIO & GIS IMPORTS WITH FALLBACK
# =========================================================
HAS_RASTERIO = False
try:
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.vrt import WarpedVRT
    import rasterio.enums

    HAS_RASTERIO = True
except Exception:
    rasterio = None

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="GeoLens | Geospatial Analysis",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# CONSTANTS & STYLES
# =========================================================
PREVIEW_MAX_WIDTH = 1800
PREVIEW_MAX_HEIGHT = 1400

st.markdown(
    """
    <style>
    :root {
        --ink: #eaf4f7;
        --muted: #8ca7b2;
        --muted-strong: #b8cbd1;
        --panel: rgba(9, 24, 34, 0.88);
        --panel-strong: #0c202d;
        --line: rgba(93, 153, 174, 0.26);
        --accent: #42d3c5;
        --accent-warm: #ffbf69;
    }
    .stApp {
        background:
            radial-gradient(circle at 70% -10%, rgba(27, 93, 107, 0.24), transparent 38%),
            radial-gradient(circle at 0% 30%, rgba(14, 58, 77, 0.18), transparent 32%),
            #061018;
        color: var(--ink);
    }
    header[data-testid="stHeader"] {
        background: rgba(6, 16, 24, 0.82);
    }
    .block-container {
        max-width: 1760px;
        padding-top: 0.7rem;
        padding-left: clamp(12px, 1.8vw, 32px);
        padding-right: clamp(12px, 1.8vw, 32px);
        padding-bottom: 1.5rem;
    }
    h3 {
        color: var(--ink);
        letter-spacing: -0.02em;
        margin-top: 0.45rem;
        margin-bottom: 0.35rem;
    }
    [data-testid="stCaptionContainer"] {
        color: var(--muted);
    }
    [data-testid="stMarkdownContainer"] p {
        color: var(--muted-strong);
    }
    [data-testid="stFileUploader"] {
        background: rgba(10, 30, 42, 0.64);
        border: 1px dashed rgba(66, 211, 197, 0.48);
        border-radius: 12px;
        padding: 4px 8px 8px;
        transition: border-color 0.2s ease, background 0.2s ease;
    }
    [data-testid="stFileUploader"]:hover {
        background: rgba(12, 42, 55, 0.78);
        border-color: var(--accent);
    }
    [data-testid="stCheckbox"] label {
        color: #c9dbe0;
        font-size: 0.86rem;
    }
    [data-testid="stCheckbox"] label:hover {
        color: #ffffff;
    }
    [data-testid="stSelectbox"] label,
    [data-testid="stSlider"] label {
        color: var(--muted-strong);
        font-size: 0.8rem;
        font-weight: 650;
    }
    div[data-baseweb="select"] > div {
        background: rgba(9, 27, 38, 0.9);
        border-color: var(--line);
        border-radius: 9px;
    }
    [data-testid="stSlider"] [role="slider"] {
        background: var(--accent);
    }
    [data-testid="stAlert"] {
        border-radius: 10px;
        border: 1px solid var(--line);
        background: rgba(10, 30, 42, 0.72);
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--line);
        border-radius: 12px;
    }
    .stButton > button {
        width: 100%;
        min-height: 42px;
        background: linear-gradient(180deg, #123143 0%, #0d2534 100%);
        color: #d9edf0;
        border: 1px solid rgba(92, 157, 175, 0.35);
        border-radius: 9px;
        font-weight: 700;
        letter-spacing: 0.01em;
        transition: transform 0.18s ease, background 0.18s ease, border-color 0.18s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(180deg, #185061 0%, #123c4d 100%);
        color: #ffffff;
        border-color: var(--accent);
        transform: translateY(-1px);
    }
    .stButton > button:focus {
        box-shadow: 0 0 0 2px rgba(66, 211, 197, 0.25);
        border-color: var(--accent);
    }
    .metric-card {
        background: linear-gradient(145deg, rgba(15, 39, 51, 0.9), rgba(7, 21, 31, 0.92));
        border: 1px solid var(--line);
        border-radius: 11px;
        padding: 11px 13px;
        margin: 8px 0;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
    }
    .metric-title {
        font-size: 10px;
        text-transform: uppercase;
        color: #7fa5b2;
        letter-spacing: 0.9px;
        font-weight: 700;
    }
    .metric-val {
        font-size: 17px;
        font-weight: 700;
        color: #effcff;
        margin-top: 2px;
    }
    .section-kicker {
        color: var(--accent);
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin: 4px 0 2px;
    }
    .help-card {
        background: rgba(11, 31, 42, 0.7);
        border: 1px solid rgba(93, 153, 174, 0.2);
        border-radius: 10px;
        padding: 11px 13px;
        color: #a8c1c9;
        font-size: 12px;
        line-height: 1.45;
        margin: 8px 0 12px;
    }
    .hero {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 18px;
        padding: 17px 22px;
        margin-bottom: 14px;
        border: 1px solid rgba(93, 153, 174, 0.3);
        border-radius: 15px;
        background:
            linear-gradient(115deg, rgba(9, 32, 44, 0.97), rgba(7, 20, 30, 0.88)),
            radial-gradient(circle at 90% 0%, rgba(66, 211, 197, 0.22), transparent 38%);
        box-shadow: 0 14px 40px rgba(0, 0, 0, 0.2);
    }
    .hero-brand {
        display: flex;
        align-items: center;
        gap: 13px;
        min-width: 0;
    }
    .hero-mark {
        width: 42px;
        height: 42px;
        display: grid;
        place-items: center;
        border-radius: 12px;
        background: linear-gradient(145deg, #43d5c6, #197e8c);
        color: #062029;
        font-size: 14px;
        font-weight: 900;
        letter-spacing: 0.08em;
        box-shadow: 0 0 24px rgba(66, 211, 197, 0.2);
    }
    .hero-title {
        color: #f1ffff;
        font-size: clamp(20px, 2.2vw, 29px);
        line-height: 1.1;
        font-weight: 800;
        letter-spacing: -0.04em;
    }
    .hero-subtitle {
        color: #9bb8c0;
        font-size: 12px;
        margin-top: 4px;
    }
    .hero-meta {
        color: #7fa7b1;
        font-size: 11px;
        text-align: right;
        white-space: nowrap;
    }
    .hero-meta strong {
        color: #d8efef;
        display: block;
        font-size: 12px;
        margin-bottom: 3px;
    }
    @media (max-width: 768px) {
        .block-container {
            padding-left: 6px;
            padding-right: 6px;
        }
        .hero {
            align-items: flex-start;
            padding: 14px;
        }
        .hero-meta {
            display: none;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.markdown(
    """
    <div class="hero">
        <div class="hero-brand">
            <div class="hero-mark">GL</div>
            <div>
                <div class="hero-title">GeoLens</div>
                <div class="hero-subtitle">Explore, inspect, and compare geospatial layers with confidence.</div>
            </div>
        </div>
        <div class="hero-meta">
            <strong>Interactive analysis workspace</strong>
            GeoTIFF&nbsp; · &nbsp;GeoJSON&nbsp; · &nbsp;2D + 3D views
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# COLORMAP UTILITIES (FOR 1-BAND SCIENTIFIC RASTERS)
# =========================================================
COLORMAPS = {
    "Viridis": [
        (0.0, (68, 1, 84)),
        (0.25, (59, 82, 139)),
        (0.5, (33, 145, 140)),
        (0.75, (94, 201, 98)),
        (1.0, (253, 231, 37))
    ],
    "NDVI / Vegetation": [
        (0.0, (165, 42, 42)),  # Bare soil / rock (brown)
        (0.25, (222, 184, 135)),  # Low vegetation / sand
        (0.5, (255, 255, 102)),  # Sparse green (yellowish)
        (0.75, (102, 204, 0)),  # Moderate vegetation (light green)
        (1.0, (0, 100, 0))  # Dense forest (deep green)
    ],
    "Spectral": [
        (0.0, (158, 1, 66)),
        (0.25, (244, 109, 67)),
        (0.5, (254, 224, 139)),
        (0.75, (102, 194, 165)),
        (1.0, (94, 79, 162))
    ],
    "Magma": [
        (0.0, (0, 0, 4)),
        (0.25, (81, 18, 124)),
        (0.5, (182, 54, 121)),
        (0.75, (251, 136, 97)),
        (1.0, (252, 253, 191))
    ],
    "Terrain": [
        (0.0, (51, 102, 153)),  # Water
        (0.2, (0, 153, 76)),  # Lowlands
        (0.5, (255, 204, 102)),  # Hills
        (0.8, (153, 102, 51)),  # Mountains
        (1.0, (255, 255, 255))  # Snow peaks
    ],
    "Greyscale": [
        (0.0, (0, 0, 0)),
        (1.0, (255, 255, 255))
    ]
}


def apply_colormap(normalized_data, valid_mask, cmap_name="Viridis"):
    """Interpolates colormap stops to produce an RGBA image array."""
    stops = COLORMAPS.get(cmap_name, COLORMAPS["Viridis"])
    h, w = normalized_data.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    norm_flat = np.nan_to_num(normalized_data.flatten(), nan=0.0)
    r_flat = np.zeros_like(norm_flat)
    g_flat = np.zeros_like(norm_flat)
    b_flat = np.zeros_like(norm_flat)

    stop_x = [s[0] for s in stops]
    stop_r = [s[1][0] for s in stops]
    stop_g = [s[1][1] for s in stops]
    stop_b = [s[1][2] for s in stops]

    r_flat = np.interp(norm_flat, stop_x, stop_r)
    g_flat = np.interp(norm_flat, stop_x, stop_g)
    b_flat = np.interp(norm_flat, stop_x, stop_b)

    rgba[:, :, 0] = r_flat.reshape((h, w)).astype(np.uint8)
    rgba[:, :, 1] = g_flat.reshape((h, w)).astype(np.uint8)
    rgba[:, :, 2] = b_flat.reshape((h, w)).astype(np.uint8)
    rgba[:, :, 3] = np.where(valid_mask, 255, 0).astype(np.uint8)
    return rgba


# =========================================================
# COORDINATE CONVERTERS (PURE PYTHON FALLBACK)
# =========================================================
def web_mercator_to_wgs84(x, y):
    lon = (x / 20037508.342789244) * 180.0
    lat = (math.atan(math.exp((y / 20037508.342789244) * math.pi)) * 4.0 - math.pi) * (180.0 / math.pi)
    return lat, lon


def utm_to_wgs84(easting, northing, zone, northern=True):
    a = 6378137.0
    f = 1 / 298.257223563
    e = math.sqrt(2 * f - f * f)
    e1sq = e * e / (1 - e * e)
    k0 = 0.9996

    x = easting - 500000.0
    y = northing if northern else northing - 10000000.0

    m = y / k0
    mu = m / (a * (1 - e ** 2 / 4 - 3 * e ** 4 / 64 - 5 * e ** 6 / 256))
    e1 = (1 - math.sqrt(1 - e ** 2)) / (1 + math.sqrt(1 - e ** 2))

    j1 = (3 * e1 / 2 - 27 * e1 ** 3 / 32)
    j2 = (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32)
    j3 = (151 * e1 ** 3 / 96)
    j4 = (1097 * e1 ** 4 / 512)

    fp = mu + j1 * math.sin(2 * mu) + j2 * math.sin(4 * mu) + j3 * math.sin(6 * mu) + j4 * math.sin(8 * mu)
    c1 = e1sq * math.cos(fp) ** 2
    t1 = math.tan(fp) ** 2
    r1 = a * (1 - e ** 2) / (1 - e ** 2 * math.sin(fp) ** 2) ** 1.5
    n1 = a / math.sqrt(1 - e ** 2 * math.sin(fp) ** 2)
    d = x / (n1 * k0)

    lat = fp - (n1 * math.tan(fp) / r1) * (
                d ** 2 / 2 - (5 + 3 * t1 + 10 * c1 - 4 * c1 ** 2 - 9 * e1sq) * d ** 4 / 24 + (
                    61 + 90 * t1 + 298 * c1 + 45 * t1 ** 2 - 252 * e1sq - 3 * c1 ** 2) * d ** 6 / 720)
    lat = math.degrees(lat)
    lon = (d - (1 + 2 * t1 + c1) * d ** 3 / 6 + (
                5 - 2 * c1 + 28 * t1 - 3 * c1 ** 2 + 8 * e1sq + 24 * t1 ** 2) * d ** 5 / 120) / math.cos(fp)
    lon_origin = (zone - 1) * 6 - 180 + 3
    lon = lon_origin + math.degrees(lon)
    return lat, lon


# =========================================================
# PURE-PYTHON GEOTIFF FALLBACK
# =========================================================
def process_tiff_fallback(file_bytes, colormap_name="Viridis"):
    """Decodes TIFF via PIL when rasterio is unavailable, extracting tags and RGBA."""
    try:
        img = Image.open(io.BytesIO(file_bytes))
        orig_w, orig_h = img.size

        scale = min(PREVIEW_MAX_WIDTH / orig_w, PREVIEW_MAX_HEIGHT / orig_h, 1.0)
        target_w = max(1, int(orig_w * scale))
        target_h = max(1, int(orig_h * scale))

        preview_img = img.resize((target_w, target_h), Image.Resampling.BILINEAR)

        tags = getattr(img, "tag_v2", {})
        scale_tag = tags.get(33550)
        tiepoint_tag = tags.get(33922)
        geokey_tag = tags.get(34735)

def transform_bounds_to_wgs84(west, south, east, north, source_crs):
    """Transform a projected bounding box to WGS84 without guessing its CRS."""
    west, south, east, north = map(float, (west, south, east, north))
    if HAS_RASTERIO:
        try:
            transformed = transform_bounds(
                source_crs, "EPSG:4326", west, south, east, north, densify_pts=21
            )
            if all(math.isfinite(float(value)) for value in transformed):
                return tuple(float(value) for value in transformed)
        except Exception:
            pass

    if HAS_PYPROJ:
        transformer = Transformer.from_crs(
            CRS.from_user_input(source_crs), CRS.from_epsg(4326), always_xy=True
        )
        sample_count = 20
        xs, ys = [], []
        for index in range(sample_count + 1):
            fraction = index / sample_count
            x = west + (east - west) * fraction
            y = south + (north - south) * fraction
            xs.extend((x, x, west, east))
            ys.extend((south, north, y, y))
        longitudes, latitudes = transformer.transform(xs, ys)
        points = [
            (float(lon), float(lat))
            for lon, lat in zip(longitudes, latitudes)
            if math.isfinite(float(lon)) and math.isfinite(float(lat))
        ]
        if not points:
            raise ValueError(f"Could not transform bounds from {source_crs} to EPSG:4326.")
        return (
            min(point[0] for point in points),
            min(point[1] for point in points),
            max(point[0] for point in points),
            max(point[1] for point in points),
        )

    crs_text = str(source_crs).upper().replace(" ", "")
    if crs_text in ("EPSG:4326", "OGC:CRS84"):
        return west, south, east, north
    raise ValueError(
        f"Cannot transform {source_crs} coordinates: Rasterio/PyProj is unavailable. Install pyproj or rasterio."
    )


        epsg = None
        if geokey_tag:
            geographic_epsg = None
            projected_epsg = None
            for i in range(0, len(geokey_tag) - 3, 4):
                key_id = geokey_tag[i]
                code = int(geokey_tag[i + 3])
                if code in (0, 32767):
                    continue
                if key_id == 2048:
                    geographic_epsg = code
                elif key_id == 3072:
                    projected_epsg = code
            epsg = projected_epsg or geographic_epsg

        if not (scale_tag and tiepoint_tag and len(tiepoint_tag) >= 6):
            raise ValueError("GeoTIFF is missing georeferencing tags. Assign a CRS and export it as a georeferenced GeoTIFF.")

        x0 = float(tiepoint_tag[3])
        y0 = float(tiepoint_tag[4])
        x1 = x0 + orig_w * float(scale_tag[0])
        y1 = y0 - orig_h * float(scale_tag[1])
        left, right = sorted((x0, x1))
        bottom, top = sorted((y0, y1))

        if epsg is None:
            if -180 <= left <= right <= 180 and -90 <= bottom <= top <= 90:
                epsg = 4326
            else:
                raise ValueError("GeoTIFF coordinates are projected but the file has no usable EPSG code. Assign the correct CRS before upload.")
        if epsg == 4326 and not (-180 <= left <= right <= 180 and -90 <= bottom <= top <= 90):
            raise ValueError("GeoTIFF coordinates exceed WGS84 ranges but its CRS metadata says EPSG:4326. Check the file CRS.")

        west, south, east, north = transform_bounds_to_wgs84(
            left, bottom, right, top, f"EPSG:{epsg}"
        )
        south = max(-85.0, min(85.0, float(south)))
        north = max(-85.0, min(85.0, float(north)))
        west = max(-180.0, min(180.0, float(west)))
        east = max(-180.0, min(180.0, float(east)))
        if south > north:
            south, north = north, south
        if west > east:
            west, east = east, west

        arr = np.array(preview_img)
        if arr.ndim == 2:
            valid = np.isfinite(arr) & (arr > 0)
            if not np.any(valid):
                valid = np.isfinite(arr)
            low = np.percentile(arr[valid], 2) if np.any(valid) else 0
            high = np.percentile(arr[valid], 98) if np.any(valid) else 1
            if high <= low: high = low + 1.0
            norm = np.clip((arr - low) / (high - low), 0.0, 1.0)
            rgba = apply_colormap(norm, valid, colormap_name)
        elif arr.shape[2] == 3:
            valid = ~((arr[:, :, 0] == 0) & (arr[:, :, 1] == 0) & (arr[:, :, 2] == 0))
            alpha = np.where(valid, 255, 0).astype(np.uint8)
            rgba = np.dstack([arr, alpha])
        elif arr.shape[2] == 4:
            rgba = arr
        else:
            rgba = np.dstack([arr[:, :, :3], np.full((target_h, target_w), 255, dtype=np.uint8)])

        out_img = Image.fromarray(rgba, "RGBA")
        buf = io.BytesIO()
        out_img.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "image": "data:image/png;base64," + encoded,
            "bounds": [[south, west], [north, east]],
            "width": orig_w,
            "height": orig_h,
            "crs": f"EPSG:{epsg}"
        }
    except Exception as e:
        return {"error": f"TIFF processing error: {str(e)}"}


# =========================================================
# HIGH-PERFORMANCE CACHED TIFF PROCESSOR
# =========================================================
@st.cache_data(show_spinner=False)
def process_tiff(file_bytes, filename, colormap_name="Viridis"):
    """Decodes, reprojects to EPSG:4326 via WarpedVRT, contrast stretches, and creates RGBA PNG."""
    if not HAS_RASTERIO:
        return process_tiff_fallback(file_bytes, colormap_name)

    try:
        with rasterio.MemoryFile(file_bytes) as memfile:
            with memfile.open() as raw_src:
                if raw_src.crs is None:
                    return process_tiff_fallback(file_bytes, colormap_name)

                # WarpedVRT ensures accurate Plate Carree (EPSG:4326) reprojection
                with WarpedVRT(raw_src, crs="EPSG:4326", resampling=rasterio.enums.Resampling.bilinear) as src:
                    scale_x = PREVIEW_MAX_WIDTH / src.width
                    scale_y = PREVIEW_MAX_HEIGHT / src.height
                    scale = min(scale_x, scale_y, 1.0)

                    width = max(1, int(src.width * scale))
                    height = max(1, int(src.height * scale))

                    # Multi-band (RGB / RGBA)
                    if src.count >= 3:
                        bands_to_read = [1, 2, 3]
                        has_alpha_band = (src.count >= 4)
                        if has_alpha_band:
                            bands_to_read.append(4)

                        data = src.read(
                            bands_to_read,
                            out_shape=(len(bands_to_read), height, width),
                            resampling=rasterio.enums.Resampling.bilinear
                        ).astype(np.float32)

                        # Mask nodata
                        if src.nodata is not None:
                            nodata_val = src.nodata
                            if np.isnan(nodata_val):
                                mask_nodata = np.isnan(data[0])
                            else:
                                mask_nodata = np.isclose(data[0], nodata_val)
                        else:
                            mask_nodata = np.zeros((height, width), dtype=bool)

                        output_rgba = np.zeros((height, width, 4), dtype=np.uint8)

                        for b_idx in range(3):
                            b_data = data[b_idx]
                            valid = np.isfinite(b_data) & (~mask_nodata)
                            if np.any(valid):
                                low = np.percentile(b_data[valid], 2)
                                high = np.percentile(b_data[valid], 98)
                                if high <= low:
                                    high = low + 1.0
                                normalized = np.clip((b_data - low) / (high - low), 0.0, 1.0) * 255
                                output_rgba[:, :, b_idx] = normalized.astype(np.uint8)
                            else:
                                output_rgba[:, :, b_idx] = 0

                        if has_alpha_band:
                            output_rgba[:, :, 3] = np.clip(data[3], 0, 255).astype(np.uint8)
                        else:
                            is_black_border = (output_rgba[:, :, 0] == 0) & (output_rgba[:, :, 1] == 0) & (
                                        output_rgba[:, :, 2] == 0)
                            valid_pixel = (~mask_nodata) & (~is_black_border) & np.isfinite(data[0])
                            output_rgba[:, :, 3] = np.where(valid_pixel, 255, 0).astype(np.uint8)

                    # Single band (DEM, NDVI, Index)
                    else:
                        data = src.read(
                            1,
                            out_shape=(height, width),
                            resampling=rasterio.enums.Resampling.bilinear
                        ).astype(np.float32)

                        if src.nodata is not None:
                            if np.isnan(src.nodata):
                                data[np.isnan(data)] = np.nan
                            else:
                                data[np.isclose(data, src.nodata)] = np.nan

                        valid = np.isfinite(data)
                        if not np.any(valid):
                            return {"error": "Raster contains no valid pixels."}

                        low = np.nanpercentile(data, 2)
                        high = np.nanpercentile(data, 98)
                        if high <= low:
                            high = low + 1.0

                        normalized = np.clip((data - low) / (high - low), 0.0, 1.0)
                        output_rgba = apply_colormap(normalized, valid, colormap_name)

                    # Encode to PNG
                    image = Image.fromarray(output_rgba, "RGBA")
                    buffer = io.BytesIO()
                    image.save(buffer, format="PNG")
                    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

                    west, south, east, north = src.bounds
                    south = max(-85.0, min(85.0, float(south)))
                    north = max(-85.0, min(85.0, float(north)))
                    west = max(-180.0, min(180.0, float(west)))
                    east = max(-180.0, min(180.0, float(east)))

                    if south > north:
                        south, north = north, south
                    if west > east:
                        west, east = east, west

                    return {
                        "image": "data:image/png;base64," + encoded,
                        "bounds": [[south, west], [north, east]],
                        "width": src.width,
                        "height": src.height,
                        "crs": "EPSG:4326"
                    }
    except Exception:
        return process_tiff_fallback(file_bytes, colormap_name)


# =========================================================
# CACHED GEOJSON PROCESSOR
# =========================================================
@st.cache_data(show_spinner=False)
def process_geojson(file_bytes, filename):
    """Parse GeoJSON, reproject declared source CRS to WGS84, and compute bounds."""
    try:
        data = json.loads(file_bytes.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("GeoJSON root must be an object.")

        declared_crs = data.get("crs")
        transformer = None
        if declared_crs:
            if isinstance(declared_crs, str):
                source_crs = declared_crs
            elif isinstance(declared_crs, dict):
                properties = declared_crs.get("properties", {}) or {}
                source_crs = properties.get("name") or properties.get("href")
                if not source_crs and properties.get("code"):
                    authority = properties.get("authority") or "EPSG"
                    source_crs = f"{authority}:{properties['code']}"
            else:
                source_crs = None
            if not source_crs:
                raise ValueError("GeoJSON declares a CRS that the app cannot read. Re-export it as EPSG:4326.")
            if not HAS_PYPROJ:
                raise ValueError("This GeoJSON declares a projected CRS; install pyproj to convert it to map coordinates.")
            transformer = Transformer.from_crs(
                CRS.from_user_input(source_crs), CRS.from_epsg(4326), always_xy=True
            )

        lats, lons = [], []

        def normalize_coordinates(node):
            if isinstance(node, list) and len(node) >= 2 and isinstance(node[0], (int, float)) and isinstance(node[1], (int, float)):
                x, y = float(node[0]), float(node[1])
                if transformer is not None:
                    lon, lat = transformer.transform(x, y, errcheck=True)
                else:
                    lon, lat = x, y
                lon, lat = float(lon), float(lat)
                if not math.isfinite(lon) or not math.isfinite(lat):
                    raise ValueError("GeoJSON contains non-finite coordinates.")
                if not (-180 <= lon <= 180 and -90 <= lat <= 90):
                    if transformer is None:
                        raise ValueError("GeoJSON coordinates are outside WGS84 longitude/latitude ranges and no CRS is declared. Re-export as EPSG:4326 or include the source CRS.")
                    raise ValueError("The declared GeoJSON CRS did not transform coordinates into valid WGS84 ranges. Check the CRS metadata.")
                lons.append(lon)
                lats.append(lat)
                return [lon, lat, *node[2:]]
            if isinstance(node, list):
                return [normalize_coordinates(child) for child in node]
            return node

        def normalize_geometry(geometry):
            if not isinstance(geometry, dict):
                return
            if "coordinates" in geometry:
                geometry["coordinates"] = normalize_coordinates(geometry["coordinates"])
            for child_geometry in geometry.get("geometries", []) or []:
                normalize_geometry(child_geometry)

        if data.get("type") == "FeatureCollection":
            features = data.get("features", [])
            if not isinstance(features, list):
                raise ValueError("GeoJSON FeatureCollection.features must be an array.")
        elif data.get("type") == "Feature":
            features = [data]
        elif data.get("type") in ("Point", "MultiPoint", "LineString", "MultiLineString", "Polygon", "MultiPolygon", "GeometryCollection"): 
            features = [{"geometry": data}]
        else:
            raise ValueError("Unsupported GeoJSON type. Use a Feature, FeatureCollection, or geometry.")

        for feature in features:
            if isinstance(feature, dict):
                normalize_geometry(feature.get("geometry"))

        if declared_crs:
            data.pop("crs", None)
        bounds = None
        if lats and lons:
            bounds = [[min(lats), min(lons)], [max(lats), max(lons)]]
        return {"data": data, "bounds": bounds, "feature_count": len(features)}
    except Exception as e:
        return {"error": f"GeoJSON parse/CRS error: {str(e)}"}

# =========================================================
# SAMPLE DATA GENERATOR (INSTANT DEMO)
# =========================================================
def generate_sample_data():
    """Creates synthetic multi-temporal satellite rasters (2020 vs 2024) and GeoJSON."""
    width, height = 320, 320
    min_lon, max_lon = -122.51, -122.34
    min_lat, max_lat = 37.71, 37.83

    def make_tiff(year, seed):
        np.random.seed(seed)
        x = np.linspace(0, 8, width)
        y = np.linspace(0, 8, height)
        xx, yy = np.meshgrid(x, y)
        terrain = np.sin(xx) * np.cos(yy)
        noise = np.random.normal(0, 0.08, (height, width))

        if year == 2020:
            r = np.clip((0.25 + 0.15 * terrain + noise) * 220, 20, 220).astype(np.uint8)
            g = np.clip((0.65 + 0.25 * terrain + noise) * 255, 60, 255).astype(np.uint8)
            b = np.clip((0.25 + 0.15 * terrain + noise) * 180, 20, 180).astype(np.uint8)
        else:
            r = np.clip((0.60 + 0.25 * terrain + noise) * 240, 40, 245).astype(np.uint8)
            g = np.clip((0.45 + 0.20 * terrain + noise) * 210, 40, 220).astype(np.uint8)
            b = np.clip((0.35 + 0.18 * terrain + noise) * 200, 30, 200).astype(np.uint8)

        cy, cx = height / 2, width / 2
        ry, rx = height * 0.46, width * 0.46
        mask = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) > 1.0
        alpha = np.full((height, width), 255, dtype=np.uint8)
        alpha[mask] = 0

        rgba = np.stack([r, g, b, alpha], axis=-1)
        img = Image.fromarray(rgba, "RGBA")
        buf = io.BytesIO()

        pixel_scale_x = (max_lon - min_lon) / width
        pixel_scale_y = (max_lat - min_lat) / height
        tiff_tags = {
            33550: (pixel_scale_x, pixel_scale_y, 0.0),
            33922: (0.0, 0.0, 0.0, min_lon, max_lat, 0.0),
            34735: (1, 1, 0, 1, 1024, 0, 1, 2, 1025, 0, 1, 1, 2048, 0, 1, 4326)
        }
        img.save(buf, format="TIFF", tiffinfo=tiff_tags)
        return buf.getvalue()

    sample_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": "Target Study Area AOI",
                    "area_km2": 42.5,
                    "sensor": "Sentinel-2 & Landsat Analysis"
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-122.48, 37.73],
                        [-122.38, 37.73],
                        [-122.36, 37.81],
                        [-122.47, 37.81],
                        [-122.48, 37.73]
                    ]]
                }
            }
        ]
    }

    return {
        "rasters": [
            {"filename": "satellite_2020.tif", "bytes": make_tiff(2020, 42)},
            {"filename": "satellite_2024.tif", "bytes": make_tiff(2024, 99)}
        ],
        "vectors": [
            {"filename": "study_boundary.geojson", "bytes": json.dumps(sample_geojson).encode("utf-8")}
        ]
    }


# =========================================================
# YEAR EXTRACTION HELPER
# =========================================================
def extract_year_or_tag(filename):
    match = re.search(r"(19|20)\d{2}", filename)
    if match:
        return int(match.group(0))
    return None


def load_folder_files(folder):
    """Load supported GIS files recursively from the configured data folder."""
    if not AUTO_LOAD_FOLDER or not folder.is_dir():
        return []

    files = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_DATA_EXTENSIONS:
            try:
                files.append({
                    "filename": path.name,
                    "bytes": path.read_bytes(),
                    "source": "folder",
                    "path": str(path)
                })
            except OSError as exc:
                st.warning(f"Could not read {path}: {exc}")
    return files


def layer_widget_key(kind, name):
    """Create a stable Streamlit key for a layer visibility checkbox."""
    digest = hashlib.sha1(f"{kind}:{name}".encode("utf-8")).hexdigest()[:12]
    return f"visible_{kind}_{digest}"


def compare_layers(available_rasters, left_label, right_label):
    """Resolve the two raster layers selected for the comparison viewer."""
    by_label = {raster["label"]: raster for raster in available_rasters}
    return by_label.get(left_label), by_label.get(right_label)


def union_bounds(bounds_list):
    """Return one bounding box containing every valid [south/west, north/east] box."""
    valid = [bounds for bounds in bounds_list if bounds and len(bounds) == 2]
    if not valid:
        return None

    return [
        [
            min(float(bounds[0][0]) for bounds in valid),
            min(float(bounds[0][1]) for bounds in valid)
        ],
        [
            max(float(bounds[1][0]) for bounds in valid),
            max(float(bounds[1][1]) for bounds in valid)
        ]
    ]


# =========================================================
# SESSION STATE INITIALIZATION
# =========================================================
if "demo_loaded" not in st.session_state:
    st.session_state["demo_loaded"] = False
if "demo_data" not in st.session_state:
    st.session_state["demo_data"] = None
if "comparison_open" not in st.session_state:
    st.session_state["comparison_open"] = False

# =========================================================
# LAYOUT COLUMNS
# =========================================================
left_col, center_col, right_col = st.columns([1.55, 5.3, 1.55], gap="small")

# =========================================================
# LEFT PANEL (DATA & LAYERS)
# =========================================================
with left_col:
    st.markdown('<div class="section-kicker">Step 1 · Get started</div>', unsafe_allow_html=True)
    st.markdown("### Add your data")
    st.markdown(
        '<div class="help-card">Upload one or more map layers, or use the demo set to see how the comparison works.</div>',
        unsafe_allow_html=True
    )
    if AUTO_LOAD_FOLDER:
        if DATA_FOLDER.is_dir():
            st.caption(f"Automatic folder: `{DATA_FOLDER}`")
        else:
            st.caption(f"Automatic folder not found: `{DATA_FOLDER}`")

    uploaded_files = st.file_uploader(
        "Choose GeoTIFF or GeoJSON files",
        type=["tif", "tiff", "geojson", "json"],
        accept_multiple_files=True,
        help="You can select several files at once. GeoTIFFs become raster layers; GeoJSON files become vector layers."
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Load demo", help="Load sample satellite layers and an area boundary"):
            st.session_state["demo_data"] = generate_sample_data()
            st.session_state["demo_loaded"] = True
            st.rerun()
    with c2:
        if st.session_state["demo_loaded"]:
            if st.button("Clear demo"):
                st.session_state["demo_loaded"] = False
                st.session_state["demo_data"] = None
                st.rerun()

    st.markdown("---")
    st.markdown('<div class="section-kicker">Step 2 · Set the scene</div>', unsafe_allow_html=True)
    st.markdown("### Map appearance")
    with st.expander("Display settings", expanded=True):
        show_rasters = st.checkbox(
            "Show raster imagery",
            value=True,
            help="Turn all raster imagery on or off without changing individual layer selections."
        )
        show_vectors = st.checkbox(
            "Show boundary layers",
            value=True,
            help="Turn all GeoJSON boundary layers on or off."
        )

        raster_opacity = st.slider(
            "Raster opacity",
            min_value=10,
            max_value=100,
            value=92,
            step=2,
            help="Lower the opacity to see the basemap underneath the imagery."
        ) / 100.0

        colormap_name = st.selectbox(
            "Single-band color palette",
            options=list(COLORMAPS.keys()),
            index=0,
            help="Used for one-band rasters such as DEM, NDVI, and other indexes."
        )

        basemap_name = st.selectbox(
            "Background map",
            options=["Satellite (Esri)", "Dark Canvas (CartoDB)", "Street (OSM)"],
            index=0
        )

# =========================================================
# PROCESS UPLOADED & DEMO FILES
# =========================================================
rasters = []
vectors = []

files_to_process = []
files_to_process.extend(load_folder_files(DATA_FOLDER))

if uploaded_files:
    for u in uploaded_files:
        files_to_process.append({
            "filename": u.name,
            "bytes": u.getvalue(),
            "source": "upload"
        })

if st.session_state.get("demo_loaded") and st.session_state.get("demo_data"):
    for r in st.session_state["demo_data"]["rasters"]:
        r = {**r, "source": "demo"}
        files_to_process.append(r)
    for v in st.session_state["demo_data"]["vectors"]:
        v = {**v, "source": "demo"}
        files_to_process.append(v)

# Prefer a manually uploaded file over demo/folder data with the same name,
# and avoid rendering duplicate demo/folder layers.
deduplicated_files = {}
source_priority = {"folder": 1, "demo": 2, "upload": 3}
for item in files_to_process:
    key = item["filename"].lower()
    current = deduplicated_files.get(key)
    if current is None or source_priority.get(item.get("source"), 0) >= source_priority.get(
        current.get("source"), 0
    ):
        deduplicated_files[key] = item

for item in files_to_process:
    if deduplicated_files.get(item["filename"].lower()) is not item:
        continue

    fname = item["filename"]
    raw = item["bytes"]
    ext = Path(fname).suffix.lower()
    year = extract_year_or_tag(fname)

    if ext in (".tif", ".tiff"):
        res = process_tiff(raw, fname, colormap_name)
        if "error" in res:
            st.sidebar.error(f"{fname}: {res['error']}")
        else:
            rasters.append({
                "name": fname,
                "label": f"{year} • {fname}" if year else fname,
                "year": year,
                "image": res["image"],
                "bounds": res["bounds"],
                "width": res["width"],
                "height": res["height"],
                "crs": res["crs"],
                "source": item.get("source", "unknown")
            })
    elif ext in (".geojson", ".json"):
        res = process_geojson(raw, fname)
        if "error" in res:
            st.sidebar.error(f"{fname}: {res['error']}")
        else:
            vectors.append({
                "name": fname,
                "label": f"{year} • {fname}" if year else fname,
                "year": year,
                "data": res["data"],
                "bounds": res["bounds"],
                "source": item.get("source", "unknown")
            })

# =========================================================
# INDIVIDUAL LAYER VISIBILITY
# =========================================================
with left_col:
    st.markdown("---")
    st.markdown('<div class="section-kicker">Step 3 · Choose layers</div>', unsafe_allow_html=True)
    st.markdown("### Layer visibility")
    raster_visibility = {}
    vector_visibility = {}

    if not rasters and not vectors:
        st.markdown(
            '<div class="help-card">Your selected layers will appear here. Start with the demo or upload files above.</div>',
            unsafe_allow_html=True
        )
    else:
        st.caption(f"{len(rasters) + len(vectors)} layer(s) available")
        select_col, clear_col = st.columns(2)
        with select_col:
            if st.button("Show all", key="show_all_layers"):
                for raster in rasters:
                    st.session_state[layer_widget_key("raster", raster["name"])] = True
                for vector in vectors:
                    st.session_state[layer_widget_key("vector", vector["name"])] = True
                st.rerun()
        with clear_col:
            if st.button("Hide all", key="hide_all_layers"):
                for raster in rasters:
                    st.session_state[layer_widget_key("raster", raster["name"])] = False
                for vector in vectors:
                    st.session_state[layer_widget_key("vector", vector["name"])] = False
                st.rerun()

        if rasters:
            st.caption("Imagery layers")
            for raster in rasters:
                key = layer_widget_key("raster", raster["name"])
                if key not in st.session_state:
                    st.session_state[key] = True
                raster_visibility[raster["name"]] = st.checkbox(
                    raster["label"],
                    key=key,
                    help=f"{raster['source'].title()} source • {raster['crs']}"
                )

        if vectors:
            st.caption("Boundary layers")
            for vector in vectors:
                key = layer_widget_key("vector", vector["name"])
                if key not in st.session_state:
                    st.session_state[key] = True
                vector_visibility[vector["name"]] = st.checkbox(
                    vector["label"],
                    key=key,
                    help=f"{vector['source'].title()} source"
                )

visible_rasters = [
    raster for raster in rasters
    if raster_visibility.get(raster["name"], True)
]
visible_vectors = [
    vector for vector in vectors
    if vector_visibility.get(vector["name"], True)
]
active_rasters = visible_rasters if show_rasters else []
active_vectors = visible_vectors if show_vectors else []

# =========================================================
# RIGHT PANEL (COMPARISON SELECTION & METADATA)
# =========================================================
with right_col:
    st.markdown('<div class="section-kicker">Compare</div>', unsafe_allow_html=True)
    st.markdown("### Compare layers")
    st.markdown(
        '<div class="help-card">Choose two visible raster layers, then drag the divider across the map to inspect change.</div>',
        unsafe_allow_html=True
    )

    left_raster = None
    right_raster = None

    if not st.session_state["comparison_open"]:
        st.caption("The comparison view stays closed until you are ready.")
        if st.button(
            "Open comparison",
            disabled=len(active_rasters) < 2,
            key="open_comparison"
        ):
            st.session_state["comparison_open"] = True
            st.rerun()
        if len(active_rasters) < 2:
            st.info("Make at least two raster layers visible to compare them.")

        if active_rasters:
            left_raster = sorted(
                active_rasters,
                key=lambda r: (r["year"] if r["year"] is not None else 9999, r["name"])
            )[0]
    else:
        st.success("Comparison view is active.")
        if st.button("Close comparison", key="close_comparison"):
            st.session_state["comparison_open"] = False
            st.rerun()

    if st.session_state["comparison_open"] and len(active_rasters) >= 2:
        rasters_sorted = sorted(
            active_rasters,
            key=lambda r: (r["year"] if r["year"] is not None else 9999, r["name"])
        )
        options = [r["label"] for r in rasters_sorted]

        left_choice = st.selectbox(
            "Base layer",
            options=options,
            index=0,
            key="comparison_left_layer"
        )
        right_choice = st.selectbox(
            "Comparison layer",
            options=options,
            index=len(options) - 1,
            key="comparison_right_layer"
        )

        left_raster, right_raster = compare_layers(
            rasters_sorted,
            left_choice,
            right_choice
        )
        if left_choice == right_choice:
            st.warning("Choose two different layers for a meaningful comparison.")

    elif st.session_state["comparison_open"]:
        if active_rasters:
            left_raster = active_rasters[0]
        st.warning("Comparison needs two visible raster layers. Turn one back on in Layer visibility.")
    elif not active_rasters:
        st.caption("No visible raster loaded. Upload files or choose **Load demo**.")

    st.markdown("---")
    st.markdown("### Current view")

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Visible Rasters</div>
            <div class="metric-val">{len(active_rasters)} / {len(rasters)}</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Visible GeoJSON Features</div>
            <div class="metric-val">{sum(len(v['data'].get('features', [])) for v in active_vectors if isinstance(v['data'], dict))}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if left_raster:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">CRS Projection</div>
                <div class="metric-val" style="font-size:13px;">{left_raster['crs']}</div>
                <div class="metric-title" style="margin-top:6px;">Dimensions</div>
                <div class="metric-val" style="font-size:13px;">{left_raster['width']} × {left_raster['height']} px</div>
            </div>
            """,
            unsafe_allow_html=True
        )

# =========================================================
# CALCULATE MAP CENTER & BOUNDS
# =========================================================
center_lat = 20.0
center_lon = 78.0
zoom = 4
initial_bounds = None

all_visible_bounds = [
    layer.get("bounds")
    for layer in [*active_rasters, *active_vectors]
    if layer.get("bounds")
]
initial_bounds = union_bounds(all_visible_bounds)

if initial_bounds:
    b = initial_bounds
    center_lat = (b[0][0] + b[1][0]) / 2.0
    center_lon = (b[0][1] + b[1][1]) / 2.0
    zoom = 9

# Data for JavaScript
left_image = left_raster["image"] if left_raster else None
left_bounds = left_raster["bounds"] if left_raster else None
right_image = right_raster["image"] if right_raster else None
right_bounds = right_raster["bounds"] if right_raster else None

vector_data = [v["data"] for v in active_vectors]

left_label_str = left_raster["label"] if left_raster else "NO DATA"
right_label_str = right_raster["label"] if right_raster else "NO DATA"

# =========================================================
# BUILD EMBEDDED HTML (LEAFLET + GLOBE.GL WITH THREE.JS)
# =========================================================
html_template = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/three@0.160.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/globe.gl@2.32.0"></script>

<style>
* { box-sizing: border-box; }
html, body {
    margin: 0; padding: 0;
    width: 100%; height: 100%;
    overflow: hidden;
    background: #02070d;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
#app {
    width: 100%; height: 100%;
    display: flex; flex-direction: column;
}
#viewSelector {
    height: 48px;
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 16px;
    background: #07111b;
    border-bottom: 1px solid #1c2b38;
    flex-shrink: 0;
}
.btn-group { display: flex; gap: 8px; align-items: center; }
.viewButton {
    border: 1px solid #243b4f;
    background: #0b1823;
    color: #aebdca;
    padding: 6px 16px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 12px;
    font-weight: 600;
    transition: all 0.2s;
}
.viewButton:hover { background: #132738; color: #ffffff; }
.viewButton.active {
    background: #17486a;
    border-color: #398ac0;
    color: #ffffff;
    box-shadow: 0 0 10px rgba(57, 138, 192, 0.4);
}
#viewer {
    position: relative;
    width: 100%;
    height: calc(100% - 48px);
    overflow: hidden;
    background: #02070d;
}
#globeLeft, #globeRight {
    position: absolute;
    inset: 0;
    width: 100%; height: 100%;
    background: #02070d;
}
#globeRight {
    clip-path: inset(0 0 0 50%);
    z-index: 20;
}
#globeLeft { z-index: 10; }

#mapLeft, #mapRight {
    position: absolute;
    inset: 0;
    width: 100%; height: 100%;
}
#mapRight {
    clip-path: inset(0 0 0 50%);
    z-index: 20;
}
#mapLeft { z-index: 10; }
#mapLeftLeaflet, #mapRightLeaflet {
    position: absolute;
    inset: 0;
    width: 100%; height: 100%;
    background: #061019;
}

#divider {
    position: absolute;
    top: 0; bottom: 0;
    left: 50%;
    width: 3px;
    transform: translateX(-50%);
    background: #00f2fe;
    z-index: 100;
    cursor: ew-resize;
    touch-action: none;
    box-shadow: 0 0 14px #00f2fe, 0 0 4px #ffffff;
}
#handle {
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    width: 40px; height: 56px;
    border-radius: 20px;
    background: #07111b;
    border: 2px solid #00f2fe;
    color: #ffffff;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    font-size: 16px;
    font-weight: bold;
    user-select: none;
    box-shadow: 0 4px 16px rgba(0,0,0,0.8), 0 0 12px rgba(0,242,254,0.4);
}
#handlePercent {
    font-size: 9px;
    color: #8edefc;
    font-weight: 700;
    margin-top: 1px;
}
.yearLabel {
    position: absolute;
    top: 12px;
    padding: 6px 12px;
    background: rgba(7, 17, 27, 0.88);
    color: #eaf2f8;
    border: 1px solid #1d3345;
    border-radius: 6px;
    z-index: 110;
    font-size: 12px;
    font-weight: 700;
    pointer-events: none;
    backdrop-filter: blur(4px);
    box-shadow: 0 2px 8px rgba(0,0,0,0.5);
    max-width: 42%;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
#leftLabel { left: 14px; border-left: 3px solid #398ac0; }
#rightLabel { right: 14px; border-right: 3px solid #00f2fe; }
#modeLabel {
    position: absolute;
    top: 12px; left: 50%;
    transform: translateX(-50%);
    z-index: 120;
    padding: 4px 10px;
    border-radius: 4px;
    background: rgba(5, 11, 18, 0.85);
    border: 1px solid #1f374a;
    color: #9ac2dc;
    font-size: 11px;
    font-weight: 600;
    pointer-events: none;
    letter-spacing: 0.5px;
}
#noDataBadge {
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    background: rgba(7, 17, 27, 0.95);
    border: 1px solid #233e54;
    padding: 18px 24px;
    border-radius: 10px;
    color: #e0edf7;
    z-index: 150;
    text-align: center;
    font-size: 14px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.8);
}
</style>
</head>

<body>
<div id="app">
    <div id="viewSelector">
        <div class="btn-group">
            <button id="mapButton" class="viewButton active" onclick="showMap()">2D map</button>
            <button id="globeButton" class="viewButton" onclick="showGlobe()">3D globe</button>
            <button class="viewButton" onclick="resetView()" title="Fit the map to the visible layers">Fit view</button>
        </div>
        <div style="font-size:12px; color:#5c788d; font-weight:600;">
            Drag the handle to reveal the comparison
        </div>
    </div>

    <div id="viewer">
        <div id="globeLeft"></div>
        <div id="globeRight"></div>

        <div id="mapLeft"><div id="mapLeftLeaflet"></div></div>
        <div id="mapRight"><div id="mapRightLeaflet"></div></div>

        <div id="leftLabel" class="yearLabel"></div>
        <div id="rightLabel" class="yearLabel"></div>
        <div id="modeLabel">2D MAP COMPARISON</div>

        <div id="divider">
            <div id="handle">
                <span>↔</span>
                <span id="handlePercent">50%</span>
            </div>
        </div>

        <div id="noDataBadge" style="display: none;">
            <div style="font-size: 24px; margin-bottom: 8px;">🛰️</div>
            <b>No Layers Selected</b><br>
            <span style="font-size: 12px; color: #8ba2b5;">Upload GeoTIFFs or click "🌟 Demo Data" in the left panel.</span>
        </div>
    </div>
</div>

<script>
// Data injected from Python
const LEFT_IMAGE = __LEFT_IMAGE__;
const LEFT_BOUNDS = __LEFT_BOUNDS__;
const RIGHT_IMAGE = __RIGHT_IMAGE__;
const RIGHT_BOUNDS = __RIGHT_BOUNDS__;
const VECTOR_DATA = __VECTOR_DATA__;
const CENTER_LAT = __CENTER_LAT__;
const CENTER_LON = __CENTER_LON__;
const INITIAL_ZOOM = __INITIAL_ZOOM__;
const VIEW_BOUNDS = __VIEW_BOUNDS__;
const LEFT_LABEL = __LEFT_LABEL__;
const RIGHT_LABEL = __RIGHT_LABEL__;
const RASTER_OPACITY = __RASTER_OPACITY__;
const BASEMAP_STYLE = __BASEMAP_STYLE__;
const COMPARISON_OPEN = __COMPARISON_OPEN__;

// DOM Elements
const divider = document.getElementById("divider");
const handlePercent = document.getElementById("handlePercent");
const globeLeft = document.getElementById("globeLeft");
const globeRight = document.getElementById("globeRight");
const mapLeft = document.getElementById("mapLeft");
const mapRight = document.getElementById("mapRight");
const modeLabel = document.getElementById("modeLabel");
const leftLabel = document.getElementById("leftLabel");
const rightLabel = document.getElementById("rightLabel");
const noDataBadge = document.getElementById("noDataBadge");

leftLabel.textContent = "LEFT: " + LEFT_LABEL;
rightLabel.textContent = "RIGHT: " + RIGHT_LABEL;

if (!LEFT_IMAGE && !RIGHT_IMAGE && (!VECTOR_DATA || VECTOR_DATA.length === 0)) {
    noDataBadge.style.display = "block";
}

// -----------------------------------------------------
// BASEMAP TILES CONFIGURATION
// -----------------------------------------------------
let tileUrl = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";
let tileAttr = "Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community";

if (BASEMAP_STYLE.includes("Dark")) {
    tileUrl = "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png";
    tileAttr = "&copy; OpenStreetMap contributors &copy; CARTO";
} else if (BASEMAP_STYLE.includes("Street")) {
    tileUrl = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";
    tileAttr = "&copy; OpenStreetMap contributors";
}

// -----------------------------------------------------
// 2D LEAFLET INITIALIZATION
// -----------------------------------------------------
let map1 = null;
let map2 = null;

function createMaps() {
    map1 = L.map("mapLeftLeaflet", {
        zoomControl: true,
        attributionControl: true,
        preferCanvas: true
    }).setView([CENTER_LAT, CENTER_LON], INITIAL_ZOOM);

    map2 = L.map("mapRightLeaflet", {
        zoomControl: false,
        attributionControl: false,
        preferCanvas: true
    }).setView([CENTER_LAT, CENTER_LON], INITIAL_ZOOM);

    // Add full basemap to both sides
    L.tileLayer(tileUrl, { maxZoom: 19, attribution: tileAttr }).addTo(map1);
    L.tileLayer(tileUrl, { maxZoom: 19 }).addTo(map2);

    // Left Raster Overlay
    if (LEFT_IMAGE && LEFT_BOUNDS) {
        L.imageOverlay(LEFT_IMAGE, LEFT_BOUNDS, { opacity: RASTER_OPACITY }).addTo(map1);
    }

    // Right Raster Overlay
    if (RIGHT_IMAGE && RIGHT_BOUNDS) {
        L.imageOverlay(RIGHT_IMAGE, RIGHT_BOUNDS, { opacity: RASTER_OPACITY }).addTo(map2);
    }

    // GeoJSON Vector Overlays
    if (VECTOR_DATA && VECTOR_DATA.length > 0) {
        const geoStyle = {
            color: "#00f2fe",
            weight: 2,
            opacity: 0.9,
            fillColor: "#00f2fe",
            fillOpacity: 0.15
        };
        VECTOR_DATA.forEach(geojson => {
            const l1 = L.geoJSON(geojson, {
                style: geoStyle,
                onEachFeature: (f, layer) => {
                    if (f.properties) {
                        layer.bindPopup("<pre style='margin:0;font-size:11px;'>" + JSON.stringify(f.properties, null, 2) + "</pre>");
                    }
                }
            }).addTo(map1);
            const l2 = L.geoJSON(geojson, { style: geoStyle }).addTo(map2);
        });
    }

    // Two-way synchronization between Map 1 and Map 2
    let syncing = false;
    function sync(source, target) {
        if (syncing) return;
        syncing = true;
        target.setView(source.getCenter(), source.getZoom(), { animate: false });
        syncing = false;
    }

    map1.on("move", () => sync(map1, map2));
    map2.on("move", () => sync(map2, map1));

    const fitBounds = VIEW_BOUNDS || LEFT_BOUNDS || RIGHT_BOUNDS;
    if (fitBounds) {
        map1.fitBounds(fitBounds, { padding: [20, 20] });
        map2.fitBounds(fitBounds, { padding: [20, 20] });
    }
}

// -----------------------------------------------------
// 3D GLOBE INITIALIZATION (WITH USER RASTER & VECTORS)
// -----------------------------------------------------
let globe1 = null;
let globe2 = null;
let globesInitialized = false;

function createGlobes() {
    if (globesInitialized) return;
    try {
        const textureUrl = "https://cdn.jsdelivr.net/npm/three-globe/example/img/earth-blue-marble.jpg";
        const bumpUrl = "https://cdn.jsdelivr.net/npm/three-globe/example/img/earth-topology.png";
        const bgUrl = "https://cdn.jsdelivr.net/npm/three-globe/example/img/night-sky.png";

        globe1 = Globe()(globeLeft)
            .globeImageUrl(textureUrl)
            .bumpImageUrl(bumpUrl)
            .backgroundImageUrl(bgUrl)
            .showAtmosphere(true)
            .atmosphereAltitude(0.15);

        globe2 = Globe()(globeRight)
            .globeImageUrl(textureUrl)
            .bumpImageUrl(bumpUrl)
            .backgroundImageUrl(bgUrl)
            .showAtmosphere(true)
            .atmosphereAltitude(0.15);

        // Project Left Raster onto Globe 1
        if (LEFT_IMAGE && LEFT_BOUNDS && window.THREE) {
            const lat = (LEFT_BOUNDS[0][0] + LEFT_BOUNDS[1][0]) / 2;
            const lng = (LEFT_BOUNDS[0][1] + LEFT_BOUNDS[1][1]) / 2;
            const width = Math.abs(LEFT_BOUNDS[1][1] - LEFT_BOUNDS[0][1]);
            const height = Math.abs(LEFT_BOUNDS[1][0] - LEFT_BOUNDS[0][0]);

            const texture = new THREE.TextureLoader().load(LEFT_IMAGE);
            const material = new THREE.MeshBasicMaterial({
                map: texture,
                transparent: true,
                opacity: RASTER_OPACITY,
                side: THREE.DoubleSide
            });

            globe1.tilesData([{ lat, lng, width, height, material }])
                .tileWidth(d => d.width)
                .tileHeight(d => d.height)
                .tileMaterial(d => d.material);
        }

        // Project Right Raster onto Globe 2
        if (RIGHT_IMAGE && RIGHT_BOUNDS && window.THREE) {
            const lat = (RIGHT_BOUNDS[0][0] + RIGHT_BOUNDS[1][0]) / 2;
            const lng = (RIGHT_BOUNDS[0][1] + RIGHT_BOUNDS[1][1]) / 2;
            const width = Math.abs(RIGHT_BOUNDS[1][1] - RIGHT_BOUNDS[0][1]);
            const height = Math.abs(RIGHT_BOUNDS[1][0] - RIGHT_BOUNDS[0][0]);

            const texture = new THREE.TextureLoader().load(RIGHT_IMAGE);
            const material = new THREE.MeshBasicMaterial({
                map: texture,
                transparent: true,
                opacity: RASTER_OPACITY,
                side: THREE.DoubleSide
            });

            globe2.tilesData([{ lat, lng, width, height, material }])
                .tileWidth(d => d.width)
                .tileHeight(d => d.height)
                .tileMaterial(d => d.material);
        }

        // Project GeoJSON Polygons onto Globes
        if (VECTOR_DATA && VECTOR_DATA.length > 0) {
            let allFeatures = [];
            VECTOR_DATA.forEach(g => {
                if (g.features) allFeatures.push(...g.features);
            });
            if (allFeatures.length > 0) {
                globe1.polygonsData(allFeatures)
                    .polygonCapColor(() => "rgba(0, 242, 254, 0.35)")
                    .polygonSideColor(() => "rgba(0, 160, 200, 0.15)")
                    .polygonStrokeColor(() => "#00f2fe")
                    .polygonAltitude(0.012)
                    .polygonLabel(d => d.properties?.name || "Feature");

                globe2.polygonsData(allFeatures)
                    .polygonCapColor(() => "rgba(0, 242, 254, 0.35)")
                    .polygonSideColor(() => "rgba(0, 160, 200, 0.15)")
                    .polygonStrokeColor(() => "#00f2fe")
                    .polygonAltitude(0.012)
                    .polygonLabel(d => d.properties?.name || "Feature");
            }
        }

        const pov = { lat: CENTER_LAT, lng: CENTER_LON, altitude: 1.8 };
        globe1.pointOfView(pov, 1000);
        globe2.pointOfView(pov, 1000);

        let syncingGlobes = false;
        globe1.controls().addEventListener("change", () => {
            if (syncingGlobes) return;
            syncingGlobes = true;
            globe2.pointOfView(globe1.pointOfView(), 0);
            syncingGlobes = false;
        });
        globe2.controls().addEventListener("change", () => {
            if (syncingGlobes) return;
            syncingGlobes = true;
            globe1.pointOfView(globe2.pointOfView(), 0);
            syncingGlobes = false;
        });

        globesInitialized = true;
    } catch (err) {
        console.error("3D Globe initialization error:", err);
    }
}

// -----------------------------------------------------
// VIEW SWITCHING (2D MAP <-> 3D GLOBE)
// -----------------------------------------------------
function applyComparisonVisibility() {
    const display = COMPARISON_OPEN ? "block" : "none";
    globeRight.style.display = display;
    mapRight.style.display = display;
    divider.style.display = display;
    rightLabel.style.display = display;
}

function showMap() {
    globeLeft.style.display = "none";
    globeRight.style.display = "none";
    mapLeft.style.display = "block";
    mapRight.style.display = "block";

    document.getElementById("mapButton").classList.add("active");
    document.getElementById("globeButton").classList.remove("active");
    modeLabel.textContent = "2D MAP COMPARISON";

    if (globe1) globe1.pauseAnimation();
    if (globe2) globe2.pauseAnimation();
    applyComparisonVisibility();

    setTimeout(() => {
        if (map1) map1.invalidateSize();
        if (map2) map2.invalidateSize();
    }, 50);
}

function showGlobe() {
    createGlobes();
    mapLeft.style.display = "none";
    mapRight.style.display = "none";
    globeLeft.style.display = "block";
    globeRight.style.display = "block";

    document.getElementById("globeButton").classList.add("active");
    document.getElementById("mapButton").classList.remove("active");
    modeLabel.textContent = "3D GLOBE COMPARISON";

    if (globe1) {
        globe1.resumeAnimation();
        globe1.width(window.innerWidth).height(window.innerHeight - 48);
    }
    if (globe2) {
        globe2.resumeAnimation();
        globe2.width(window.innerWidth).height(window.innerHeight - 48);
    }
    applyComparisonVisibility();
}

function resetView() {
    const fitBounds = VIEW_BOUNDS || LEFT_BOUNDS || RIGHT_BOUNDS;
    if (map1 && fitBounds) {
        map1.fitBounds(fitBounds, { padding: [30, 30] });
        if (map2) map2.fitBounds(fitBounds, { padding: [30, 30] });
    } else if (map1) {
        map1.setView([CENTER_LAT, CENTER_LON], INITIAL_ZOOM);
        if (map2) map2.setView([CENTER_LAT, CENTER_LON], INITIAL_ZOOM);
    }
    if (globe1) {
        globe1.pointOfView({ lat: CENTER_LAT, lng: CENTER_LON, altitude: 1.8 }, 1000);
        globe2.pointOfView({ lat: CENTER_LAT, lng: CENTER_LON, altitude: 1.8 }, 1000);
    }
}

// -----------------------------------------------------
// SLIDER DRAG LOGIC (TOUCH & POINTER EVENT RESILIENT)
// -----------------------------------------------------
let isDragging = false;

function setSliderPosition(clientX) {
    const rect = document.getElementById("viewer").getBoundingClientRect();
    let pct = ((clientX - rect.left) / rect.width) * 100;
    pct = Math.max(0, Math.min(100, pct));
    const rounded = Math.round(pct);

    divider.style.left = pct + "%";
    handlePercent.textContent = rounded + "%";

    globeRight.style.clipPath = `inset(0 0 0 ${pct}%)`;
    mapRight.style.clipPath = `inset(0 0 0 ${pct}%)`;
}

divider.addEventListener("pointerdown", (e) => {
    isDragging = true;
    divider.setPointerCapture(e.pointerId);
    e.preventDefault();
});

window.addEventListener("pointermove", (e) => {
    if (!isDragging) return;
    setSliderPosition(e.clientX);
});

window.addEventListener("pointerup", () => { isDragging = false; });
window.addEventListener("pointercancel", () => { isDragging = false; });

// -----------------------------------------------------
// BOOTSTRAP
// -----------------------------------------------------
createMaps();
showMap(); // Start in 2D Map mode for instant interactive response
applyComparisonVisibility();

window.addEventListener("resize", () => {
    if (map1) map1.invalidateSize();
    if (map2) map2.invalidateSize();
    if (globe1) globe1.width(window.innerWidth).height(window.innerHeight - 48);
    if (globe2) globe2.width(window.innerWidth).height(window.innerHeight - 48);
});
</script>
</body>
</html>
"""

# Replace placeholders safely with JSON strings
html = html_template.replace("__LEFT_IMAGE__", json.dumps(left_image))
html = html.replace("__LEFT_BOUNDS__", json.dumps(left_bounds))
html = html.replace("__RIGHT_IMAGE__", json.dumps(right_image))
html = html.replace("__RIGHT_BOUNDS__", json.dumps(right_bounds))
html = html.replace("__VECTOR_DATA__", json.dumps(vector_data))
html = html.replace("__CENTER_LAT__", str(center_lat))
html = html.replace("__CENTER_LON__", str(center_lon))
html = html.replace("__INITIAL_ZOOM__", str(zoom))
html = html.replace("__VIEW_BOUNDS__", json.dumps(initial_bounds))
html = html.replace("__LEFT_LABEL__", json.dumps(left_label_str))
html = html.replace("__RIGHT_LABEL__", json.dumps(right_label_str))
html = html.replace("__RASTER_OPACITY__", str(raster_opacity))
html = html.replace("__BASEMAP_STYLE__", json.dumps(basemap_name))
html = html.replace(
    "__COMPARISON_OPEN__",
    json.dumps(st.session_state["comparison_open"])
)

# =========================================================
# RENDER CENTER VIEWER
# =========================================================
with center_col:
    if not rasters and not vectors:
        st.markdown(
            '<div class="help-card"><strong>Start here:</strong> upload a GeoTIFF or GeoJSON file, or choose <strong>Load demo</strong> on the left. Your interactive map will appear in this space.</div>',
            unsafe_allow_html=True
        )
    else:
        visible_count = len(active_rasters) + len(active_vectors)
        comparison_hint = "Comparison view is ready." if len(active_rasters) >= 2 else "Select another raster to enable comparison."
        st.caption(f"{visible_count} visible layer(s) · {comparison_hint}")
    components.html(html, height=790, scrolling=False)