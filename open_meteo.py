"""
open_meteo.py - fetch daily weather from Open-Meteo.

Open-Meteo (https://open-meteo.com) is free for non-commercial use and needs no
API key. This module has two functions, both returning the same tidy shape:

    from open_meteo import open_meteo, open_meteo_history

    open_meteo("London", 5)                              # next few days (forecast)
    open_meteo_history("London", "2026-01-01", "2026-01-07")  # a past date range

Both return a pandas DataFrame with one row per day and the columns:

    date, day_of_week, temp, tempmax, humidity, precip, windspeed,
    cloudcover, visibility, solarenergy, uvindex

The additional fields line up with model_l. Visibility is converted to
kilometres and solar energy is expressed in MJ/m2 to match the training data.

The forecast reaches about 7 days ahead. For any date in the past (for example
the first week of January 2026) use open_meteo_history, which reads Open-Meteo's
historical archive.
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from functools import lru_cache

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DEFAULT_VISIBILITY_KM = 24.0  # training-data mean; archive visibility can be absent
MONTHLY_UV_CLIMATOLOGY = {
    1: 1.505, 2: 2.575, 3: 4.075, 4: 5.792, 5: 6.366, 6: 6.586,
    7: 6.237, 8: 5.734, 9: 4.508, 10: 2.871, 11: 1.769, 12: 1.126,
}
FORECAST_CACHE_SECONDS = 6 * 60 * 60
HISTORY_CACHE_SECONDS = 30 * 24 * 60 * 60
CACHE_DIR = Path(
    os.environ.get(
        "OPEN_METEO_CACHE_DIR",
        Path(tempfile.gettempdir()) / "london-bike-weather-cache",
    )
)

# Forecast API daily field  ->  the column name our model uses
DAILY_FIELDS = {
    "temperature_2m_mean": "temp",
    "temperature_2m_max": "tempmax",
    "relative_humidity_2m_mean": "humidity",
    "precipitation_sum": "precip",
    "wind_speed_10m_mean": "windspeed",
    "cloud_cover_mean": "cloudcover",
    "shortwave_radiation_sum": "solarenergy",
    "uv_index_max": "uvindex",
}

# The archive API has no daily means, so we pull these hourly fields and
# aggregate them ourselves (mean for most, sum for precipitation).
HOURLY_FIELDS = {
    "temperature_2m": "temp",
    "relative_humidity_2m": "humidity",
    "precipitation": "precip",
    "wind_speed_10m": "windspeed",
    "cloud_cover": "cloudcover",
    "visibility": "visibility",
    "shortwave_radiation": "solarenergy",
}

def _session():
    """Build a session with short, bounded retries for transient API failures."""
    retry = Retry(
        total=2,
        connect=2,
        read=2,
        status=2,
        backoff_factor=0.4,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        # A public API can return a very long Retry-After value. Keep the
        # dashboard responsive and use cached/local data after bounded retries.
        respect_retry_after_header=False,
        raise_on_status=False,
    )
    session = requests.Session()
    session.headers.update({"User-Agent": "LondonBikeDemandDashboard/1.0"})
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


HTTP = _session()


def _cache_path(kind, key):
    digest = hashlib.sha256(f"{kind}:{key}".encode("utf-8")).hexdigest()[:24]
    return CACHE_DIR / f"{kind}-{digest}.json"


def _write_cache(kind, key, frame):
    """Persist successful responses so process restarts do not waste API calls."""
    path = _cache_path(kind, key)
    rows = frame.assign(
        date=pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    )
    rows = rows.astype(object).where(pd.notna(rows), None)
    payload = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "location": frame.attrs.get("location"),
        "rows": rows.to_dict("records"),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
        temporary.replace(path)
    except (OSError, TypeError, ValueError):
        # A read-only/ephemeral filesystem should not take down the dashboard.
        pass


def _read_cache(kind, key, max_age_seconds, allow_stale=False):
    path = _cache_path(kind, key)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        saved_at = datetime.fromisoformat(payload["saved_at"])
        age = (datetime.now(timezone.utc) - saved_at).total_seconds()
        if age > max_age_seconds and not allow_stale:
            return None
        frame = pd.DataFrame(payload["rows"])
        frame["date"] = pd.to_datetime(frame["date"])
        frame.attrs.update(
            {
                "location": payload.get("location"),
                "source": "stale cache" if age > max_age_seconds else "cache",
                "cache_age_hours": max(0, age) / 3600,
            }
        )
        return frame
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _get_json(url, params, timeout):
    response = HTTP.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


@lru_cache(maxsize=32)
def geocode(location):
    """Turn a place name into (latitude, longitude, label) using Open-Meteo."""
    payload = _get_json(
        GEOCODE_URL,
        {"name": location, "count": 1, "language": "en"},
        timeout=15,
    )
    results = payload.get("results")
    if not results:
        raise ValueError(f"Open-Meteo could not find a location called {location!r}.")
    top = results[0]
    label = ", ".join(p for p in [top.get("name"), top.get("country")] if p)
    return top["latitude"], top["longitude"], label


def open_meteo(location="London", days_to_forecast=5):
    """Return a daily weather forecast for a location as a tidy DataFrame.

    Args:
        location (str): a place name, e.g. "London" or "Paris".
        days_to_forecast (int): number of days ahead, from 1 to 7
            (Open-Meteo's forecast does not go beyond 7 days).

    Returns:
        pandas.DataFrame with daily weather fields used by the dashboard.
        The resolved place name is stored in df.attrs["location"].
    """
    days_to_forecast = int(days_to_forecast)
    if not 1 <= days_to_forecast <= 7:
        raise ValueError("days_to_forecast must be between 1 and 7 (Open-Meteo's forecast limit).")

    location = (location or "London").strip()
    cache_key = f"{location.casefold()}:{days_to_forecast}"
    cached = _read_cache("forecast", cache_key, FORECAST_CACHE_SECONDS)
    if cached is not None:
        return cached

    try:
        lat, lon, label = geocode(location)

        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": ",".join(DAILY_FIELDS),
            "hourly": "visibility",
            "forecast_days": days_to_forecast,
            "timezone": "auto",
            "wind_speed_unit": "kmh",   # matches the training data units
        }
        payload = _get_json(FORECAST_URL, params, timeout=15)
        daily = payload["daily"]

        df = pd.DataFrame({col: daily[field] for field, col in DAILY_FIELDS.items()})
        df.insert(0, "date", pd.to_datetime(daily["time"]))
        hourly = payload["hourly"]
        visibility = pd.DataFrame({
            "date": pd.to_datetime(hourly["time"]).normalize(),
            "visibility": hourly["visibility"],
        }).groupby("date", as_index=False)["visibility"].mean()
        visibility["visibility"] = (
            pd.to_numeric(visibility["visibility"], errors="coerce") / 1000
        )
        visibility["visibility"] = visibility["visibility"].fillna(DEFAULT_VISIBILITY_KM)
        df = df.merge(visibility, on="date", how="left")
        df["visibility"] = df["visibility"].fillna(DEFAULT_VISIBILITY_KM)
        df.insert(1, "day_of_week", df["date"].dt.strftime("%a"))
        df["month"] = df["date"].dt.month
        df.attrs.update({"location": label, "source": "Open-Meteo live"})
        _write_cache("forecast", cache_key, df)
        return df
    except (requests.RequestException, KeyError, TypeError, ValueError):
        stale = _read_cache(
            "forecast", cache_key, FORECAST_CACHE_SECONDS, allow_stale=True
        )
        if stale is not None:
            return stale
        raise


def open_meteo_history(location, start_date, end_date):
    """Return daily weather for a past date range from Open-Meteo's archive.

    Use this for dates the forecast cannot reach, such as the first week of
    January 2026. The archive has no daily means, so we pull the hourly values
    and aggregate them to one row per day here.

    Args:
        location (str): a place name, e.g. "London".
        start_date (str): first day, "YYYY-MM-DD".
        end_date (str): last day, "YYYY-MM-DD".

    Returns:
        pandas.DataFrame with the same columns as open_meteo().
    """
    location = (location or "London").strip()
    cache_key = f"{location.casefold()}:{start_date}:{end_date}"
    cached = _read_cache("history", cache_key, HISTORY_CACHE_SECONDS)
    if cached is not None:
        return cached

    try:
        lat, lon, label = geocode(location)

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(HOURLY_FIELDS),
            "timezone": "auto",
            "wind_speed_unit": "kmh",   # matches the training data units
        }
        payload = _get_json(ARCHIVE_URL, params, timeout=30)
        hourly = payload["hourly"]

        hf = pd.DataFrame({col: hourly[field] for field, col in HOURLY_FIELDS.items()})
        hf["date"] = pd.to_datetime(hourly["time"]).normalize()

        # Aggregate hours to days: mean for levels, sum for precipitation
        daily = hf.groupby("date").agg(
            temp=("temp", "mean"),
            tempmax=("temp", "max"),
            humidity=("humidity", "mean"),
            precip=("precip", "sum"),
            windspeed=("windspeed", "mean"),
            cloudcover=("cloudcover", "mean"),
            visibility=("visibility", "mean"),
            solarenergy=("solarenergy", lambda values: values.sum() * 0.0036),
        ).reset_index()
        daily["visibility"] = pd.to_numeric(
            daily["visibility"], errors="coerce"
        ) / 1000
        daily["visibility"] = daily["visibility"].fillna(DEFAULT_VISIBILITY_KM)

        daily.insert(1, "day_of_week", daily["date"].dt.strftime("%a"))
        daily["month"] = daily["date"].dt.month
        # The archive endpoint does not expose historical UV index. Use the
        # training dataset's monthly UV climatology as a transparent fallback.
        daily["uvindex"] = daily["month"].map(MONTHLY_UV_CLIMATOLOGY)
        daily.attrs.update({"location": label, "source": "Open-Meteo archive"})
        _write_cache("history", cache_key, daily)
        return daily
    except (requests.RequestException, KeyError, TypeError, ValueError):
        stale = _read_cache(
            "history", cache_key, HISTORY_CACHE_SECONDS, allow_stale=True
        )
        if stale is not None:
            return stale
        raise


if __name__ == "__main__":
    # Quick manual checks (need internet)
    print("Forecast (next 5 days):")
    print(open_meteo("London", 5).to_string(index=False))
    print("\nHistory (first week of January 2026):")
    print(open_meteo_history("London", "2026-01-01", "2026-01-07").to_string(index=False))
