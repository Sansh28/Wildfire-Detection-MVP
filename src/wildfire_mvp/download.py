from __future__ import annotations

from datetime import date
from datetime import datetime, timedelta
import io
import os
from pathlib import Path
import zipfile

import csv
import math

import numpy as np


def download_firms_csv(
    output_csv: Path,
    region: str,
    start_date: date,
    end_date: date,
    api_key_env: str = "FIRMS_API_KEY",
) -> None:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("requests is required to download FIRMS data") from exc

    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing FIRMS API key in environment variable {api_key_env}")

    if end_date < start_date:
        raise ValueError("end_date must be >= start_date")

    # FIRMS area endpoint expects a bbox string and a day-range in [1..5].
    region_aliases = {
        "western_us": "-125,24,-66,49",
        "canada": "-141,41,-52,84",
    }
    area = region_aliases.get(region.strip().lower(), region)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    base_url = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

    merged_rows: list[dict] = []
    seen_keys: set[tuple] = set()
    cursor = start_date
    while cursor <= end_date:
        chunk_end = min(cursor + timedelta(days=4), end_date)
        day_range = (chunk_end - cursor).days + 1
        url = f"{base_url}/{api_key}/VIIRS_SNPP_NRT/{area}/{day_range}/{chunk_end.isoformat()}"
        response = requests.get(url, timeout=120)
        response.raise_for_status()

        decoded = response.content.decode("utf-8")
        reader = csv.DictReader(decoded.splitlines())
        for row in reader:
            dedupe_key = (
                row.get("latitude", ""),
                row.get("longitude", ""),
                row.get("acq_date", ""),
                row.get("acq_time", ""),
            )
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            merged_rows.append(row)

        cursor = chunk_end + timedelta(days=1)

    if not merged_rows:
        output_csv.write_text("", encoding="utf-8")
        return

    fieldnames = list(merged_rows[0].keys())
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)


def _require_earth_engine():
    try:
        import ee
    except ImportError as exc:
        raise RuntimeError(
            "earthengine-api is required for Sentinel-2 extraction. Install it with: pip install earthengine-api"
        ) from exc
    return ee


def _require_rasterio():
    try:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.io import MemoryFile
    except ImportError as exc:
        raise RuntimeError("rasterio is required for Sentinel-2 extraction") from exc
    return rasterio, MemoryFile, Resampling


def initialize_earth_engine(
    service_account_json: str = "", project: str = "", service_account_email_env: str = "GOOGLE_SERVICE_ACCOUNT_EMAIL"
) -> None:
    ee = _require_earth_engine()
    if service_account_json:
        credentials_path = Path(service_account_json).expanduser().resolve()
        if not credentials_path.exists():
            raise RuntimeError(f"Service account JSON not found: {credentials_path}")
        service_account = os.environ.get(service_account_email_env, "")
        if not service_account:
            raise RuntimeError(
                f"Set {service_account_email_env} when using service_account_json for Earth Engine initialization"
            )
        credentials = ee.ServiceAccountCredentials(service_account, str(credentials_path))
        ee.Initialize(credentials=credentials, project=project or None)
        return

    # Falls back to ADC or previous `earthengine authenticate` login.
    ee.Initialize(project=project or None)


def _event_region(lon: float, lat: float, image_size: int, scale_m: int):
    ee = _require_earth_engine()
    half_extent_m = (image_size * scale_m) / 2.0
    dlat = half_extent_m / 111_320.0
    cos_lat = max(math.cos(math.radians(lat)), 1e-6)
    dlon = half_extent_m / (111_320.0 * cos_lat)
    return ee.Geometry.Rectangle([lon - dlon, lat - dlat, lon + dlon, lat + dlat], proj=None, geodesic=False)


def _pick_s2_image(
    lon: float,
    lat: float,
    target_date: date,
    collection: str,
    max_cloud_cover: float,
    search_window_days: int,
    region,
    min_clear_fraction: float,
    max_candidates: int,
):
    ee = _require_earth_engine()
    point = ee.Geometry.Point([lon, lat])
    start = (target_date - timedelta(days=search_window_days)).isoformat()
    end = (target_date + timedelta(days=search_window_days + 1)).isoformat()
    coll = (
        ee.ImageCollection(collection)
        .filterBounds(point)
        .filterDate(start, end)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud_cover * 100.0))
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )
    candidate_count = int(coll.size().getInfo())
    if candidate_count == 0:
        coll = (
            ee.ImageCollection(collection)
            .filterBounds(point)
            .filterDate(start, end)
            .sort("CLOUDY_PIXEL_PERCENTAGE")
        )
        candidate_count = int(coll.size().getInfo())
        if candidate_count == 0:
            return None

    candidate_limit = max(1, min(max_candidates, candidate_count))
    candidates = coll.limit(candidate_limit).toList(candidate_limit)
    for idx in range(candidate_limit):
        image = ee.Image(candidates.get(idx))
        qa = image.select("QA60")
        clear_mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
        clear_fraction = clear_mask.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=20,
            maxPixels=1_000_000,
            bestEffort=True,
        ).get("QA60")
        if clear_fraction is None:
            continue
        if float(clear_fraction.getInfo()) >= min_clear_fraction:
            return image.updateMask(clear_mask)
    # Best-effort fallback: keep the least-cloudy available scene instead of
    # skipping the event entirely when no scene satisfies the clear-fraction gate.
    return ee.Image(candidates.get(0))


def _download_ee_geotiff(image, region, bands: list[str], image_size: int, scale_m: int) -> np.ndarray:
    _, MemoryFile, Resampling = _require_rasterio()
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("requests is required for Sentinel-2 extraction") from exc

    region_geojson = region.getInfo()
    url = image.select(bands).getDownloadURL(
        {
            "region": region_geojson,
            "scale": scale_m,
            "format": "GEO_TIFF",
            "crs": "EPSG:4326",
        }
    )
    response = requests.get(url, timeout=180)
    response.raise_for_status()
    payload = response.content

    geotiff_bytes: bytes | None = None
    buffer = io.BytesIO(payload)
    if zipfile.is_zipfile(buffer):
        with zipfile.ZipFile(buffer) as archive:
            tif_names = [name for name in archive.namelist() if name.lower().endswith((".tif", ".tiff"))]
            if not tif_names:
                raise RuntimeError("Earth Engine download zip did not contain a GeoTIFF")
            geotiff_bytes = archive.read(tif_names[0])
    else:
        geotiff_bytes = payload

    if geotiff_bytes is None:
        raise RuntimeError("Unable to read Earth Engine GeoTIFF payload")

    with MemoryFile(geotiff_bytes) as memfile:
        with memfile.open() as dataset:
            data = dataset.read(
                out_shape=(dataset.count, image_size, image_size),
                resampling=Resampling.bilinear,
            ).astype(np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(data / 10000.0, 0.0, 1.0)


def _read_event_rows(events_csv: Path) -> list[dict]:
    with events_csv.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _normalize_ee_band_name(name: str) -> str:
    token = name.strip().upper()
    if token.startswith("B") and len(token) == 3 and token[1:].isdigit():
        # Earth Engine S2 bands are B1..B12 (without zero padding).
        return f"B{int(token[1:])}"
    return token


def extract_sentinel_patches(
    events_csv: Path,
    output_dir: Path,
    provider: str,
    collection: str,
    service_account_json: str = "",
    project: str = "",
    bands: list[str] | None = None,
    frame_offsets_days: list[int] | None = None,
    image_size: int = 128,
    max_cloud_cover: float = 0.2,
    scale_m: int = 20,
    search_window_days: int = 3,
    service_account_email_env: str = "GOOGLE_SERVICE_ACCOUNT_EMAIL",
    min_clear_fraction: float = 0.7,
    max_candidates: int = 5,
) -> None:
    if provider != "earth_engine":
        raise ValueError(f"Unsupported Sentinel provider: {provider}")

    active_bands = [_normalize_ee_band_name(item) for item in (bands or ["B04", "B03", "B02", "B08"])]
    offsets = sorted(frame_offsets_days or [-10, -5, 0])
    if len(offsets) != 3:
        raise ValueError("MVP expects exactly 3 frame offsets")

    initialize_earth_engine(
        service_account_json=service_account_json,
        project=project,
        service_account_email_env=service_account_email_env,
    )
    rows = _read_event_rows(events_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    extracted = 0
    skipped = 0

    for row in rows:
        event_id = row.get("event_id", "")
        if not event_id:
            skipped += 1
            continue
        lat = float(row["lat"])
        lon = float(row["lon"])
        event_date = datetime.fromisoformat(row["date"]).date()

        region = _event_region(lon=lon, lat=lat, image_size=image_size, scale_m=scale_m)
        frame_arrays: list[np.ndarray] = []
        missing_frame = False

        for offset in offsets:
            target = event_date + timedelta(days=offset)
            image = _pick_s2_image(
                lon=lon,
                lat=lat,
                target_date=target,
                collection=collection,
                max_cloud_cover=max_cloud_cover,
                search_window_days=search_window_days,
                region=region,
                min_clear_fraction=min_clear_fraction,
                max_candidates=max_candidates,
            )
            if image is None:
                missing_frame = True
                break
            frame = _download_ee_geotiff(
                image=image,
                region=region,
                bands=active_bands,
                image_size=image_size,
                scale_m=scale_m,
            )
            frame_arrays.append(frame)

        if missing_frame:
            skipped += 1
            continue

        sequence = np.stack(frame_arrays, axis=0)
        np.save(output_dir / f"seq_{event_id}.npy", sequence)
        extracted += 1

    print({"extracted": extracted, "skipped": skipped, "output_dir": str(output_dir)})
