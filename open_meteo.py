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

import requests
import pandas as pd

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DEFAULT_VISIBILITY_KM = 24.0  # training-data mean; archive visibility can be absent
MONTHLY_UV_CLIMATOLOGY = {
    1: 1.505, 2: 2.575, 3: 4.075, 4: 5.792, 5: 6.366, 6: 6.586,
    7: 6.237, 8: 5.734, 9: 4.508, 10: 2.871, 11: 1.769, 12: 1.126,
}

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


def geocode(location):
    """Turn a place name into (latitude, longitude, label) using Open-Meteo."""
    resp = requests.get(
        GEOCODE_URL,
        params={"name": location, "count": 1, "language": "en"},
        timeout=15,
    )
    resp.raise_for_status()
    results = resp.json().get("results")
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
    resp = requests.get(FORECAST_URL, params=params, timeout=15)
    resp.raise_for_status()
    daily = resp.json()["daily"]

    df = pd.DataFrame({col: daily[field] for field, col in DAILY_FIELDS.items()})
    df.insert(0, "date", pd.to_datetime(daily["time"]))
    hourly = resp.json()["hourly"]
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
    df.attrs["location"] = label
    return df


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
    resp = requests.get(ARCHIVE_URL, params=params, timeout=30)
    resp.raise_for_status()
    hourly = resp.json()["hourly"]

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
    daily.attrs["location"] = label
    return daily


if __name__ == "__main__":
    # Quick manual checks (need internet)
    print("Forecast (next 5 days):")
    print(open_meteo("London", 5).to_string(index=False))
    print("\nHistory (first week of January 2026):")
    print(open_meteo_history("London", "2026-01-01", "2026-01-07").to_string(index=False))
