"""Interactive London bike demand dashboard for the AM01 group assignment."""

from pathlib import Path
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, callback, dash_table, dcc, html

from open_meteo import open_meteo, open_meteo_history


BASE_DIR = Path(__file__).resolve().parent
DATA_URL = (
    "https://raw.githubusercontent.com/kostis-christodoulou/"
    "am01-code-sep2026/main/data/london_bikes.csv"
)
MODEL_VARIABLES = ["temp", "precip", "windspeed", "visibility", "uvindex"]
VARIABLE_LABELS = {
    "temp": "Temperature (°C)",
    "tempmax": "Maximum temperature (°C)",
    "precip": "Precipitation (mm)",
    "windspeed": "Wind speed (km/h)",
    "visibility": "Visibility (km)",
    "solarenergy": "Solar energy (MJ/m²)",
    "uvindex": "UV index",
    "humidity": "Humidity (%)",
    "cloudcover": "Cloud cover (%)",
}
DAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTH_ORDER = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


def load_data():
    """Load and prepare the course dataset."""
    local_path = BASE_DIR.parent / "data" / "london_bikes.csv"
    try:
        frame = pd.read_csv(local_path)
    except (FileNotFoundError, OSError):
        frame = pd.read_csv(DATA_URL)
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame = frame[frame["date"] >= pd.Timestamp("2014-01-01", tz="UTC")].copy()
    frame["weekend_label"] = frame["day_of_week"].isin(["Sat", "Sun"]).map(
        {True: "Weekend", False: "Weekday"}
    )
    return frame


def load_coefficients():
    """Read the exported linear-model coefficients."""
    table = pd.read_csv(BASE_DIR / "model_coefficients.csv")
    return dict(zip(table["term"], table["coefficient"]))


BIKES = load_data()
COEFFICIENTS = load_coefficients()


def predict_hires(frame):
    """Apply model_l1 coefficients to weather and calendar rows."""
    result = pd.Series(COEFFICIENTS["Intercept"], index=frame.index, dtype=float)
    for variable in MODEL_VARIABLES:
        result = result + COEFFICIENTS[variable] * pd.to_numeric(
            frame[variable], errors="coerce"
        )
    result = result + frame["day_of_week"].map(
        lambda day: COEFFICIENTS.get(f"day_{day}", 0.0)
    )
    months = (
        frame["month"]
        if "month" in frame
        else pd.to_datetime(frame["date"]).dt.month
    )
    result = result + months.map(
        lambda month: COEFFICIENTS.get(f"month_{int(month)}", 0.0)
    )
    return result.clip(lower=0)


BIKES["model_prediction"] = predict_hires(BIKES)
BIKES["residual"] = BIKES["bikes_hired"] - BIKES["model_prediction"]


def climatology_weather(dates):
    """Build transparent date-specific weather estimates from London history."""
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).tz_localize(None).normalize()
    target = pd.DataFrame({"date": dates})
    target["month"] = target["date"].dt.month
    target["day"] = target["date"].dt.day

    historical = BIKES.assign(day=pd.to_datetime(BIKES["date"]).dt.day)
    climatology = (
        historical.groupby(["month", "day"], as_index=False)[MODEL_VARIABLES]
        .median()
    )
    result = target.merge(climatology, on=["month", "day"], how="left")

    # Month/global medians cover leap days or an unexpectedly sparse local file.
    monthly = historical.groupby("month")[MODEL_VARIABLES].median()
    for variable in MODEL_VARIABLES:
        result[variable] = pd.to_numeric(result[variable], errors="coerce")
        result[variable] = result[variable].fillna(
            result["month"].map(monthly[variable])
        ).fillna(historical[variable].median())
    result["day_of_week"] = result["date"].dt.strftime("%a")
    first_year = historical["date"].dt.year.min()
    last_year = historical["date"].dt.year.max()
    result.attrs.update(
        {
            "location": "London",
            "source": f"London dataset climatology ({first_year}–{last_year})",
        }
    )
    return result[["date", "day_of_week", *MODEL_VARIABLES, "month"]]


def usable_weather(fetcher, fallback_dates):
    """Fetch weather, filling gaps or replacing failures with local climatology."""
    try:
        frame = fetcher()
        if frame.empty:
            raise ValueError("weather response contained no rows")
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
        local = climatology_weather(frame["date"])
        filled = []
        for variable in MODEL_VARIABLES:
            values = (
                pd.to_numeric(frame[variable], errors="coerce")
                if variable in frame
                else pd.Series(float("nan"), index=frame.index)
            )
            missing = values.isna()
            if missing.any():
                replacements = pd.Series(
                    local[variable].to_numpy(), index=frame.index
                )
                values = values.fillna(replacements)
                filled.append(variable)
            frame[variable] = values
        if "day_of_week" not in frame:
            frame["day_of_week"] = frame["date"].dt.strftime("%a")
        if "month" not in frame:
            frame["month"] = frame["date"].dt.month
        if filled:
            frame.attrs["note"] = (
                "Local climatology filled missing " + ", ".join(filled) + "."
            )
        return frame, None
    except Exception as error:
        fallback = climatology_weather(fallback_dates)
        status_code = getattr(getattr(error, "response", None), "status_code", None)
        reason = f"HTTP {status_code}" if status_code else type(error).__name__
        return fallback, reason


def blank_figure(message):
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 16, "color": "#9aabc0"},
    )
    figure.update_layout(
        template="plotly_dark",
        height=380,
        paper_bgcolor="#0a1422",
        plot_bgcolor="#0a1422",
    )
    return figure


def style_figure(figure, height=420):
    figure.update_layout(
        template="plotly_dark",
        height=height,
        margin={"l": 48, "r": 24, "t": 62, "b": 44},
        paper_bgcolor="#0a1422",
        plot_bgcolor="#0a1422",
        font={
            "family": "Inter, -apple-system, BlinkMacSystemFont, sans-serif",
            "color": "#b7c5d6",
            "size": 12,
        },
        legend_title_text="",
        hoverlabel={"bgcolor": "#10293a", "font_color": "#f4f8fc"},
        hovermode="x unified",
    )
    figure.update_xaxes(gridcolor="rgba(148,163,184,.10)", zeroline=False)
    figure.update_yaxes(gridcolor="rgba(148,163,184,.10)", zeroline=False)
    return figure


def card(title, value, note="", trend=None, accent="cyan"):
    return html.Div(
        [
            html.Div(
                [
                    html.Div(title, className="metric-label"),
                    html.Span("●", className=f"metric-dot {accent}"),
                ],
                className="metric-heading",
            ),
            html.Div(value, className="metric-value"),
            html.Div(
                [
                    html.Span(trend, className="trend-pill") if trend else None,
                    html.Span(note, className="metric-note"),
                ],
                className="metric-footer",
            ),
        ],
        className="metric-card",
    )


def section_header(eyebrow, title, description):
    return html.Div(
        [
            html.Div(eyebrow, className="eyebrow"),
            html.H2(title),
            html.P(description, className="section-description"),
        ],
        className="section-header",
    )


def overview_page():
    return html.Div(
        [
            section_header(
                "EXECUTIVE PULSE",
                "Network demand at a glance",
                "A decision layer for operations, capacity planning, and weather risk.",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Reporting window"),
                            dcc.DatePickerRange(
                                id="overview-date-range",
                                min_date_allowed=BIKES["date"].min().date(),
                                max_date_allowed=BIKES["date"].max().date(),
                                start_date=BIKES["date"].max().date()
                                - pd.Timedelta(days=1095),
                                end_date=BIKES["date"].max().date(),
                            ),
                        ],
                        className="control",
                    ),
                    html.Div(
                        [
                            html.Label("Season"),
                            dcc.Dropdown(
                                id="overview-season",
                                options=[
                                    {"label": season, "value": season}
                                    for season in ["Winter", "Spring", "Summer", "Autumn"]
                                ],
                                value=["Winter", "Spring", "Summer", "Autumn"],
                                multi=True,
                            ),
                        ],
                        className="control",
                    ),
                    html.Div(
                        [
                            html.Div("DATA STATUS", className="filter-caption"),
                            html.Div(
                                [
                                    html.Span(className="live-dot"),
                                    " Historical dataset connected",
                                ],
                                className="data-status",
                            ),
                        ],
                        className="filter-status",
                    ),
                ],
                className="filter-bar",
            ),
            html.Div(
                [
                    card(
                        "TOTAL HIRES",
                        html.Span(id="overview-total"),
                        "selected period",
                        accent="cyan",
                    ),
                    card(
                        "AVG DAILY DEMAND",
                        html.Span(id="overview-average"),
                        "hires / day",
                        accent="teal",
                    ),
                    card(
                        "PEAK DAY",
                        html.Span(id="overview-peak"),
                        html.Span(id="overview-peak-date"),
                        accent="amber",
                    ),
                    card(
                        "WEEKEND GAP",
                        html.Span(id="overview-weekend-gap"),
                        "vs weekdays",
                        accent="rose",
                    ),
                ],
                className="metric-grid",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div("DEMAND TREND", className="chart-kicker"),
                                    html.Div(
                                        "Monthly network hires",
                                        className="chart-subtitle",
                                    ),
                                ],
                                className="chart-heading",
                            ),
                            dcc.Graph(
                                id="overview-demand-trend",
                                config={"displayModeBar": False},
                            ),
                        ],
                        className="chart-card span-two",
                    ),
                    html.Div(
                        [
                            html.Div("AUTO INSIGHT", className="insight-label"),
                            html.Div(id="overview-insight", className="insight-copy"),
                            html.Div(
                                "Model L1 • refreshed from selected period",
                                className="insight-meta",
                            ),
                        ],
                        className="insight-card",
                    ),
                ],
                className="overview-grid",
            ),
            html.Div(
                [
                    dcc.Graph(
                        id="overview-heatmap",
                        className="chart-card",
                        config={"displayModeBar": False},
                    ),
                    dcc.Graph(
                        id="overview-season-chart",
                        className="chart-card",
                        config={"displayModeBar": False},
                    ),
                    dcc.Graph(
                        id="overview-drivers",
                        className="chart-card",
                        config={"displayModeBar": False},
                    ),
                ],
                className="three-column",
            ),
        ],
        className="page",
    )


def explore_page():
    return html.Div(
        [
            section_header(
                "DEMAND EXPLORER",
                "Find the patterns behind ridership",
                "Slice weather relationships and calendar behaviour without changing the model.",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Weather variable"),
                            dcc.Dropdown(
                                id="explore-variable",
                                options=[
                                    {"label": label, "value": value}
                                    for value, label in VARIABLE_LABELS.items()
                                    if value in BIKES.columns
                                ],
                                value="temp",
                                clearable=False,
                            ),
                        ],
                        className="control",
                    ),
                    html.Div(
                        [
                            html.Label("Colour points by"),
                            dcc.RadioItems(
                                id="explore-colour",
                                options=[
                                    {"label": " Weekend", "value": "weekend_label"},
                                    {"label": " Season", "value": "season_name"},
                                ],
                                value="season_name",
                                inline=True,
                            ),
                        ],
                        className="control",
                    ),
                ],
                className="control-row",
            ),
            html.Div(
                [
                    dcc.Graph(id="weather-scatter", className="chart-card"),
                    dcc.Graph(id="weekday-bar", className="chart-card"),
                ],
                className="two-column",
            ),
        ],
        className="page",
    )


def model_page():
    return html.Div(
        [
            section_header(
                "MODEL OBSERVATORY",
                "Model L1 through time",
                "Track fit, residual behaviour, and where the model systematically misses demand.",
            ),
            html.Div(
                [
                    card("Adjusted R²", "0.600", "Model L1"),
                    card("Residual SE", "5,812", "daily hires"),
                    card("Training days", f"{len(BIKES):,}", "2014 onward"),
                    card(
                        "Mean absolute error",
                        f"{BIKES['residual'].abs().mean():,.0f}",
                        "daily hires",
                    ),
                ],
                className="metric-grid",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Date range"),
                            dcc.DatePickerRange(
                                id="model-date-range",
                                min_date_allowed=BIKES["date"].min().date(),
                                max_date_allowed=BIKES["date"].max().date(),
                                start_date=BIKES["date"].max().date()
                                - pd.Timedelta(days=730),
                                end_date=BIKES["date"].max().date(),
                            ),
                        ],
                        className="control",
                    ),
                    html.Div(
                        [
                            html.Label("Time aggregation"),
                            dcc.Dropdown(
                                id="model-frequency",
                                options=[
                                    {"label": "Weekly", "value": "W"},
                                    {"label": "Monthly", "value": "ME"},
                                    {"label": "Quarterly", "value": "QE"},
                                    {"label": "Yearly", "value": "YE"},
                                ],
                                value="W",
                                clearable=False,
                            ),
                        ],
                        className="control",
                    ),
                ],
                className="control-row",
            ),
            dcc.Graph(id="model-time-series", className="chart-card"),
            html.Div(
                [
                    dcc.Graph(id="residual-time-series", className="chart-card"),
                    dcc.Graph(id="residual-distribution", className="chart-card"),
                ],
                className="two-column",
            ),
        ],
        className="page",
    )


def prediction_page():
    controls = []
    defaults = {
        "temp": 14,
        "precip": 0,
        "windspeed": 18,
        "visibility": 24,
        "uvindex": 4,
    }
    ranges = {
        "temp": (-10, 35, 1),
        "precip": (0, 50, 1),
        "windspeed": (0, 80, 1),
        "visibility": (0, 65, 1),
        "uvindex": (0, 12, 0.5),
    }
    for variable in MODEL_VARIABLES:
        minimum, maximum, step = ranges[variable]
        controls.append(
            html.Div(
                [
                    html.Label(VARIABLE_LABELS[variable]),
                    dcc.Slider(
                        minimum,
                        maximum,
                        step,
                        value=defaults[variable],
                        id=f"scenario-{variable}",
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                ],
                className="slider-control",
            )
        )

    return html.Div(
        [
            section_header(
                "SCENARIO STUDIO",
                "Turn weather into an operating plan",
                "Stress-test Model L1 manually, then compare against live and historical weather.",
            ),
            html.Div("CUSTOM INPUT", className="subsection-label"),
            html.Div(
                [
                    html.Div(
                        controls
                        + [
                            html.Div(
                                [
                                    html.Label("Scenario date"),
                                    dcc.DatePickerSingle(
                                        id="scenario-date",
                                        date="2026-01-05",
                                    ),
                                ],
                                className="control",
                            )
                        ],
                        className="scenario-controls",
                    ),
                    dcc.Graph(id="scenario-gauge", className="chart-card"),
                    dcc.Graph(id="contribution-chart", className="chart-card"),
                ],
                className="scenario-grid",
            ),
            html.Div(id="scenario-summary", className="decision-strip"),
            html.Div("WEATHER-BASED FORECAST", className="subsection-label"),
            html.P(
                "When the historical archive omits visibility or UV index, the model uses "
                "transparent training-data climatology fallbacks.",
                className="muted",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Location"),
                            dcc.Input(
                                id="forecast-location",
                                value="London",
                                type="text",
                                debounce=True,
                            ),
                        ],
                        className="control",
                    ),
                    html.Button(
                        "Refresh weather",
                        id="refresh-weather",
                        n_clicks=0,
                        className="primary-button",
                    ),
                    html.Div(id="weather-status", className="status"),
                ],
                className="control-row",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H4("1–7 January 2026"),
                            dcc.Graph(id="history-prediction"),
                            dash_table.DataTable(
                                id="history-table",
                                page_size=7,
                                style_table={"overflowX": "auto"},
                                style_header={
                                    "backgroundColor": "#101d2d",
                                    "color": "#5eead4",
                                    "fontWeight": "bold",
                                },
                                style_cell={
                                    "padding": "7px",
                                    "textAlign": "right",
                                    "backgroundColor": "#0a1422",
                                    "color": "#f4f8fc",
                                    "borderColor": "#26384d",
                                },
                            ),
                        ],
                        className="chart-card",
                    ),
                    html.Div(
                        [
                            html.H4("Next five days"),
                            dcc.Graph(id="forecast-prediction"),
                            dash_table.DataTable(
                                id="forecast-table",
                                page_size=5,
                                style_table={"overflowX": "auto"},
                                style_header={
                                    "backgroundColor": "#101d2d",
                                    "color": "#5eead4",
                                    "fontWeight": "bold",
                                },
                                style_cell={
                                    "padding": "7px",
                                    "textAlign": "right",
                                    "backgroundColor": "#0a1422",
                                    "color": "#f4f8fc",
                                    "borderColor": "#26384d",
                                },
                            ),
                        ],
                        className="chart-card",
                    ),
                ],
                className="two-column",
            ),
        ],
        className="page",
    )


def raw_data_page():
    display = BIKES.copy()
    display["date"] = display["date"].dt.strftime("%Y-%m-%d")
    columns = [
        "date",
        "bikes_hired",
        "model_prediction",
        "temp",
        "precip",
        "windspeed",
        "visibility",
        "uvindex",
        "day_of_week",
        "month",
        "season_name",
    ]
    display["model_prediction"] = display["model_prediction"].round()
    return html.Div(
        [
            section_header(
                "DATA WORKBENCH",
                "Audit every observation",
                f"{len(display):,} records from 2014 onward with native filtering and multi-column sorting.",
            ),
            dash_table.DataTable(
                data=display[columns].to_dict("records"),
                columns=[{"name": column, "id": column} for column in columns],
                page_size=20,
                filter_action="native",
                sort_action="native",
                sort_mode="multi",
                fixed_rows={"headers": True},
                style_table={"height": "650px", "overflowY": "auto"},
                style_header={
                    "backgroundColor": "#101d2d",
                    "color": "#5eead4",
                    "fontWeight": "bold",
                },
                style_filter={
                    "backgroundColor": "#101d2d",
                    "color": "#f4f8fc",
                    "borderColor": "#31445c",
                },
                style_cell={
                    "padding": "8px",
                    "fontFamily": "SFMono-Regular, Consolas, monospace",
                    "fontSize": "0.8rem",
                    "textAlign": "right",
                    "minWidth": "95px",
                    "backgroundColor": "#0a1422",
                    "color": "#f4f8fc",
                    "borderColor": "#26384d",
                },
                style_data_conditional=[
                    {
                        "if": {"row_index": "odd"},
                        "backgroundColor": "#0d1928",
                    }
                ],
            ),
        ],
        className="page",
    )


app = Dash(__name__, title="London Bike Demand")
server = app.server

app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            * { box-sizing: border-box; }
            :root {
                --ink: #f4f8fc;
                --muted: #9aabc0;
                --line: rgba(148, 163, 184, .18);
                --surface: #0a1422;
                --surface-2: #101d2d;
                --canvas: #050b13;
                --teal: #2dd4bf;
                --cyan: #22d3ee;
                --positive: #a3e635;
                --amber: #fbbf24;
                --danger: #fb7185;
            }
            body { margin: 0; color: var(--ink);
                   background-color: var(--canvas);
                   background-image:
                     linear-gradient(rgba(34,211,238,.025) 1px, transparent 1px),
                     linear-gradient(90deg, rgba(34,211,238,.025) 1px, transparent 1px),
                     radial-gradient(circle at 75% 0%, rgba(45,212,191,.08), transparent 34%);
                   background-size: 32px 32px, 32px 32px, auto;
                   font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
            .app-shell { max-width: 1600px; margin: auto; padding: 18px 28px 38px; }
            .hero { position: relative; overflow: hidden; padding: 27px 34px; color: white;
                    border: 1px solid rgba(34,211,238,.20); border-radius: 16px;
                    background:
                    radial-gradient(circle at 85% -50%, rgba(45,212,191,.18), transparent 42%),
                    linear-gradient(120deg, #07111d 0%, #0c1a29 62%, #0a2730 100%);
                    box-shadow: 0 22px 55px rgba(0, 0, 0, .32), inset 0 1px rgba(255,255,255,.03);
                    display: flex; align-items: center; justify-content: space-between; gap: 20px; }
            .hero:after { content: ""; position: absolute; right: 5%; top: 18px;
                          width: 170px; height: 70px; opacity: .25;
                          background: repeating-linear-gradient(90deg, transparent 0 9px,
                          var(--teal) 10px 11px); transform: skewX(-18deg); }
            .hero h1 { margin: 0 0 7px; font-size: 1.85rem; letter-spacing: -.035em; }
            .hero p { margin: 0; color: #a9b8ca; font-size: .92rem; }
            .hero .eyebrow { color: var(--teal); margin-bottom: 6px; }
            .hero-badge { position: relative; z-index: 1; padding: 9px 13px;
                          border: 1px solid rgba(45,212,191,.28); border-radius: 999px;
                          background: rgba(45,212,191,.08); color: #c8fff6; backdrop-filter: blur(8px);
                          font-size: .68rem; font-weight: 800; letter-spacing: .1em; white-space: nowrap; }
            .system-rail { display: flex; align-items: center; gap: 0; margin-top: 10px;
                           min-height: 35px; border: 1px solid var(--line); border-radius: 9px;
                           background: rgba(8,16,30,.82); overflow: hidden;
                           font-family: SFMono-Regular, Consolas, monospace; }
            .rail-item { padding: 9px 16px; border-right: 1px solid var(--line);
                         color: #b5c5d8; font-size: .66rem; letter-spacing: .045em; }
            .rail-label { margin-right: 8px; color: #708198; }
            .rail-live { margin-left: auto; color: var(--teal); border-right: 0; }
            .rail-live .live-dot { width: 6px; height: 6px; }
            .page { padding: 28px 0 0; }
            .section-header { margin-bottom: 20px; }
            .section-header h2 { margin: 4px 0 5px; font-size: 1.55rem; letter-spacing: -.03em; }
            .eyebrow, .chart-kicker, .subsection-label, .insight-label, .filter-caption {
                color: var(--teal); font-size: .68rem; font-weight: 800; letter-spacing: .14em;
            }
            .section-description { margin: 0; color: var(--muted); font-size: .9rem; }
            .subsection-label { margin: 24px 0 12px; }
            .control-row, .filter-bar { display: flex; gap: 16px; align-items: end;
                                       flex-wrap: wrap; margin-bottom: 18px; }
            .filter-bar { background: rgba(11,20,36,.84); border: 1px solid var(--line);
                          border-radius: 13px; box-shadow: inset 0 1px rgba(255,255,255,.025);
                          padding: 14px 16px; }
            .control { min-width: 230px; flex: 1; }
            label { display: block; margin-bottom: 7px; color: #a9bad0; font-size: .76rem;
                    font-weight: 750; letter-spacing: .02em; }
            .filter-status { min-width: 250px; padding: 4px 10px 8px; }
            .data-status { margin-top: 7px; color: #b9c9db; font-size: .82rem; font-weight: 650; }
            .live-dot { display: inline-block; width: 8px; height: 8px; margin-right: 6px;
                        border-radius: 50%; background: var(--positive);
                        box-shadow: 0 0 0 4px rgba(163,230,53,.10), 0 0 13px rgba(163,230,53,.45); }
            .two-column { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
                          gap: 16px; margin-bottom: 16px; }
            .three-column { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
                            gap: 16px; margin-bottom: 16px; }
            .overview-grid { display: grid; grid-template-columns: minmax(0, 2.15fr) minmax(270px, .85fr);
                             gap: 16px; margin-bottom: 16px; }
            .chart-card, .metric-card, .insight-card, .scenario-controls {
                background: var(--surface); border: 1px solid var(--line); border-radius: 14px;
                box-shadow: 0 10px 28px rgba(0, 0, 0, .22), inset 0 1px rgba(255,255,255,.025); }
            .chart-card { padding: 10px; }
            .metric-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
                           gap: 14px; margin-bottom: 16px; }
            .metric-card { position: relative; overflow: hidden; padding: 17px 18px 15px; }
            .metric-card:after { content: ""; position: absolute; left: 0; bottom: 0;
                                 width: 100%; height: 2px;
                                 background: linear-gradient(90deg, var(--teal), var(--cyan), transparent); }
            .metric-heading, .metric-footer { display: flex; align-items: center;
                                              justify-content: space-between; gap: 8px; }
            .metric-label { color: #8fa1b8; font-size: .68rem; font-weight: 800; letter-spacing: .09em; }
            .metric-dot { font-size: .62rem; }
            .metric-dot.cyan { color: var(--cyan); } .metric-dot.teal { color: var(--teal); }
            .metric-dot.amber { color: var(--amber); } .metric-dot.rose { color: var(--danger); }
            .metric-value { margin: 9px 0 8px; font-size: 1.75rem; line-height: 1;
                            color: #f2f8ff; font-family: SFMono-Regular, Consolas, monospace;
                            font-weight: 760; letter-spacing: -.055em; }
            .metric-note { color: #8fa1b8; font-size: .72rem; }
            .trend-pill { padding: 3px 7px; color: var(--positive); background: rgba(163,230,53,.08);
                          border-radius: 999px; font-size: .68rem; font-weight: 750; }
            .chart-heading { padding: 8px 12px 0; }
            .chart-subtitle { margin-top: 3px; color: #71839c; font-size: .78rem; }
            .insight-card { min-height: 420px; padding: 27px; color: white;
                            background: linear-gradient(145deg, #0c1728, #10243a);
                            border-color: rgba(49,197,244,.16); display: flex; flex-direction: column;
                            justify-content: center; }
            .insight-label { color: var(--teal); }
            .insight-copy { margin: 18px 0 24px; font-size: 1.25rem; line-height: 1.55;
                            font-weight: 650; letter-spacing: -.02em; }
            .insight-meta { color: #71839c; font-size: .72rem; }
            .scenario-grid { display: grid; grid-template-columns: 1.15fr .8fr 1.1fr;
                             gap: 16px; margin-bottom: 28px; align-items: stretch; }
            .scenario-controls { padding: 22px; }
            .slider-control { margin-bottom: 25px; }
            .decision-strip { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
                              gap: 12px; margin: -12px 0 30px; }
            .decision-card { padding: 15px 17px; border: 1px solid var(--line);
                             border-radius: 12px; background: linear-gradient(145deg, #0b1424, #0f1b2e); }
            .decision-label { color: #7c8aa0; font-size: .62rem; font-weight: 800;
                              letter-spacing: .1em; }
            .decision-value { margin: 7px 0 4px; color: #f2f8ff; font-size: 1.35rem;
                              font-family: SFMono-Regular, Consolas, monospace;
                              font-weight: 760; letter-spacing: -.035em; }
            .decision-value.small { font-size: 1rem; }
            .decision-note { color: #71839c; font-size: .7rem; }
            .primary-button { border: 0; border-radius: 9px; padding: 11px 18px;
                              background: linear-gradient(120deg, var(--teal), var(--cyan));
                              color: #03120f; font-weight: 800; cursor: pointer;
                              box-shadow: 0 7px 18px rgba(45,212,191,.16); }
            .primary-button:hover { transform: translateY(-1px); filter: brightness(1.08); }
            .primary-button:focus-visible { outline: 3px solid rgba(34,211,238,.42);
                                            outline-offset: 2px; }
            .status { color: #8fa2ba; padding: 10px; font-size: .8rem; }
            .muted { color: var(--muted); font-size: .82rem; }
            .tab-container { margin-top: 14px; border: 1px solid var(--line);
                             border-radius: 12px 12px 0 0; background: rgba(11,20,36,.72);
                             backdrop-filter: blur(14px); overflow: hidden; }
            .tab-container .tab { border: 0 !important; border-bottom: 2px solid transparent !important;
                                  background: transparent !important; padding: 15px 18px !important;
                                  color: #71839c; font-size: .72rem; font-weight: 800;
                                  letter-spacing: .07em; }
            .tab-container .tab--selected { color: var(--teal) !important;
                                            border-bottom-color: var(--teal) !important;
                                            background: rgba(45,212,191,.055) !important; }
            .Select-control, .Select-menu-outer, .DateRangePickerInput,
            .SingleDatePickerInput, input[type="text"], input[type="number"] {
                background: #0f1b2e !important; border-color: #25344a !important;
                color: #f4f8fc !important;
            }
            input::placeholder { color: #91a2b8 !important; opacity: 1; }
            .Select-value-label, .Select-placeholder, .DateInput_input,
            .DateInput_input__focused, .DateInput, .DateRangePickerInput_arrow,
            .SingleDatePickerInput_calendarIcon_svg { color: #f4f8fc !important;
                background: #0f1b2e !important; }
            .Select-control:hover, .DateRangePickerInput:hover, .SingleDatePickerInput:hover,
            input[type="text"]:hover, input[type="number"]:hover {
                border-color: #3c536c !important;
            }
            .Select-control:focus-within, .DateRangePickerInput:focus-within,
            .SingleDatePickerInput:focus-within, input[type="text"]:focus,
            input[type="number"]:focus {
                border-color: var(--cyan) !important;
                box-shadow: 0 0 0 3px rgba(34,211,238,.14) !important;
                outline: none;
            }
            .Select-menu-outer { z-index: 1000 !important; }
            .VirtualizedSelectOption { background: #0f1b2e; color: #d9e6f5; }
            .VirtualizedSelectFocusedOption { background: #173044; color: #d6fffa; }
            .dash-dropdown, .dash-dropdown-trigger,
            .dash-datepicker-input-wrapper, .dash-datepicker-input {
                background: #0f1b2e !important; border-color: #25344a !important;
                color: #d9e6f5 !important;
            }
            .dash-dropdown-value-item { color: #d9e6f5 !important; }
            .dash-dropdown-value-count { color: #a9bad0 !important;
                                         background: #16263c !important; }
            .dash-dropdown-clear, .dash-dropdown-trigger-icon,
            .dash-datepicker-range-arrow { color: #7f91aa !important; }
            .dash-dropdown-content { background: #0f1b2e !important;
                                     border-color: #25344a !important; color: #d9e6f5 !important; }
            .rc-slider-rail { background-color: #26384d !important; }
            .rc-slider-track {
                background: linear-gradient(90deg, var(--cyan), var(--teal)) !important;
            }
            .rc-slider-handle {
                width: 18px; height: 18px; margin-top: -7px;
                border: 3px solid var(--teal) !important; background: #07131f !important;
                box-shadow: 0 0 0 3px rgba(45,212,191,.16) !important;
                opacity: 1 !important;
            }
            .rc-slider-handle:hover, .rc-slider-handle:focus {
                border-color: #67e8f9 !important;
                box-shadow: 0 0 0 5px rgba(34,211,238,.20) !important;
            }
            .rc-slider-tooltip-inner {
                min-width: 46px; padding: 7px 9px;
                border: 1px solid rgba(45,212,191,.45);
                border-radius: 7px; background: #0d2630 !important; color: #f4fffd !important;
                box-shadow: 0 6px 18px rgba(0,0,0,.38);
                font-family: SFMono-Regular, Consolas, monospace; font-weight: 800;
                font-size: .78rem; line-height: 1;
            }
            .rc-slider-tooltip-arrow { border-bottom-color: #0d2630 !important; }
            .dash-slider-track { background: #26384d !important; }
            .dash-slider-range {
                background: linear-gradient(90deg, var(--cyan), var(--teal)) !important;
            }
            .dash-slider-thumb {
                background: #07131f !important;
                border: 3px solid var(--teal) !important;
                box-shadow: 0 0 0 3px rgba(45,212,191,.16) !important;
            }
            .dash-slider-thumb:hover, .dash-slider-thumb:focus-visible {
                border-color: #67e8f9 !important;
                box-shadow: 0 0 0 5px rgba(34,211,238,.20) !important;
                outline: none;
            }
            .slider-control .dash-slider-tooltip {
                min-width: 40px; padding: 7px 9px !important;
                border: 1px solid #22d3ee;
                border-radius: 7px !important;
                background: #ccfbf1 !important; color: #062033 !important;
                fill: #ccfbf1 !important;
                box-shadow: 0 6px 18px rgba(0,0,0,.38);
                font-family: SFMono-Regular, Consolas, monospace;
                font-size: .78rem; font-weight: 800; line-height: 1;
            }
            .slider-control .dash-range-slider-input {
                background: #d9fbff !important; color: #062033 !important;
                border: 1px solid #22d3ee !important; border-radius: 6px !important;
                font-family: SFMono-Regular, Consolas, monospace;
                font-weight: 700; text-align: center;
            }
            .slider-control .dash-range-slider-input:hover {
                background: #ccfbf1 !important; border-color: #2dd4bf !important;
            }
            .slider-control .dash-range-slider-input:focus,
            .slider-control .dash-range-slider-input:focus-visible {
                background: #ecfeff !important; color: #031725 !important;
                border-color: #67e8f9 !important;
                box-shadow: 0 0 0 3px rgba(34,211,238,.32) !important;
                outline: 2px solid #0e7490 !important; outline-offset: 1px;
            }
            .slider-control .dash-range-slider-input::selection {
                background: #0e7490 !important; color: #ecfeff !important;
            }
            .DayPicker, .CalendarMonth, .CalendarMonthGrid,
            .DayPicker_transitionContainer { background: #0f1b2e !important; }
            .CalendarMonth_caption, .DayPicker_weekHeader, .CalendarDay {
                color: #d9e6f5 !important;
            }
            .CalendarDay { background: #0f1b2e !important; border-color: #26384d !important; }
            .CalendarDay:hover { background: #173044 !important; color: #f4fffd !important; }
            .CalendarDay__selected, .CalendarDay__selected:hover {
                background: #0f766e !important; border-color: var(--teal) !important;
                color: #ffffff !important;
            }
            .CalendarDay__selected_span { background: #134e4a !important;
                                          border-color: #26736d !important; }
            .DayPickerNavigation_button { background: #15263a !important;
                                          border-color: #31445c !important; }
            .dash-table-container { border-radius: 10px; overflow: hidden; }
            .dash-table-container input { background: #101d2d !important;
                                          color: #f4f8fc !important;
                                          border-color: #31445c !important; }
            @media (max-width: 1050px) {
                .two-column, .three-column, .scenario-grid, .overview-grid {
                    grid-template-columns: 1fr;
                }
                .decision-strip { grid-template-columns: 1fr; }
                .metric-grid { grid-template-columns: repeat(2, 1fr); }
            }
            @media (max-width: 620px) {
                .app-shell { padding: 10px 12px 28px; }
                .hero { padding: 22px; }
                .metric-grid { grid-template-columns: 1fr; }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>{%config%}{%scripts%}{%renderer%}</footer>
    </body>
</html>
"""

app.layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [
                        html.Div("TfL ANALYTICS • OPERATIONS", className="eyebrow"),
                        html.H1("London Bike Intelligence"),
                        html.P(
                            "Demand command centre for planning, forecasting, and model governance."
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Span(className="live-dot"),
                        "MODEL L1 · LIVE",
                    ],
                    className="hero-badge",
                ),
            ],
            className="hero",
        ),
        html.Div(
            [
                html.Div(
                    [html.Span("DATASET", className="rail-label"), "4,383 OBS"],
                    className="rail-item",
                ),
                html.Div(
                    [html.Span("MODEL", className="rail-label"), "L1 / OLS"],
                    className="rail-item",
                ),
                html.Div(
                    [html.Span("ADJ R²", className="rail-label"), "0.600"],
                    className="rail-item",
                ),
                html.Div(
                    [html.Span("SOURCE", className="rail-label"), "OPEN-METEO"],
                    className="rail-item",
                ),
                html.Div(
                    [html.Span(className="live-dot"), "SYSTEMS NOMINAL"],
                    className="rail-item rail-live",
                ),
            ],
            className="system-rail",
        ),
        dcc.Tabs(
            [
                dcc.Tab(label="OVERVIEW", children=overview_page()),
                dcc.Tab(label="EXPLORE", children=explore_page()),
                dcc.Tab(label="MODEL LAB", children=model_page()),
                dcc.Tab(label="SCENARIO STUDIO", children=prediction_page()),
                dcc.Tab(label="DATA WORKBENCH", children=raw_data_page()),
            ],
            mobile_breakpoint=0,
            className="tab-container",
        ),
    ],
    className="app-shell",
)


@callback(
    Output("overview-total", "children"),
    Output("overview-average", "children"),
    Output("overview-peak", "children"),
    Output("overview-peak-date", "children"),
    Output("overview-weekend-gap", "children"),
    Output("overview-demand-trend", "figure"),
    Output("overview-heatmap", "figure"),
    Output("overview-season-chart", "figure"),
    Output("overview-drivers", "figure"),
    Output("overview-insight", "children"),
    Input("overview-date-range", "start_date"),
    Input("overview-date-range", "end_date"),
    Input("overview-season", "value"),
)
def update_overview(start_date, end_date, seasons):
    selected = BIKES[
        BIKES["date"].between(
            pd.Timestamp(start_date, tz="UTC"),
            pd.Timestamp(end_date, tz="UTC"),
        )
    ].copy()
    if seasons:
        selected = selected[selected["season_name"].isin(seasons)]
    if selected.empty:
        empty = blank_figure("No data for the selected filters")
        return "—", "—", "—", "No peak", "—", empty, empty, empty, empty, "Adjust the filters to restore data."

    total = selected["bikes_hired"].sum()
    average = selected["bikes_hired"].mean()
    peak_row = selected.loc[selected["bikes_hired"].idxmax()]
    weekday_average = selected.loc[
        selected["weekend_label"] == "Weekday", "bikes_hired"
    ].mean()
    weekend_average = selected.loc[
        selected["weekend_label"] == "Weekend", "bikes_hired"
    ].mean()
    weekend_gap = weekend_average / weekday_average - 1

    monthly = (
        selected.set_index("date")["bikes_hired"]
        .resample("ME")
        .sum()
        .reset_index()
    )
    monthly["rolling"] = monthly["bikes_hired"].rolling(3, min_periods=1).mean()
    trend = go.Figure()
    trend.add_trace(
        go.Bar(
            x=monthly["date"],
            y=monthly["bikes_hired"],
            name="Monthly hires",
            marker_color="rgba(49, 197, 244, .24)",
            hovertemplate="%{x|%b %Y}<br>%{y:,.0f} hires<extra></extra>",
        )
    )
    trend.add_trace(
        go.Scatter(
            x=monthly["date"],
            y=monthly["rolling"],
            name="3-month signal",
            mode="lines",
            line={"color": "#2dd4bf", "width": 3},
            hovertemplate="%{x|%b %Y}<br>%{y:,.0f} rolling hires<extra></extra>",
        )
    )
    trend.update_layout(showlegend=True, barmode="overlay")

    heatmap_data = (
        selected.groupby(["month_name", "day_of_week"], observed=True)["bikes_hired"]
        .mean()
        .unstack()
        .reindex(index=MONTH_ORDER, columns=DAY_ORDER)
    )
    heatmap = go.Figure(
        go.Heatmap(
            z=heatmap_data.values,
            x=DAY_ORDER,
            y=MONTH_ORDER,
            colorscale=[
                [0, "#111d30"],
                [0.45, "#0e7490"],
                [1, "#2dd4bf"],
            ],
            colorbar={"title": "Avg hires", "thickness": 10},
            hovertemplate="%{y} · %{x}<br>%{z:,.0f} avg hires<extra></extra>",
        )
    )
    heatmap.update_layout(title="Demand intensity matrix")

    season_order = ["Winter", "Spring", "Summer", "Autumn"]
    season_summary = (
        selected.groupby("season_name", observed=True)["bikes_hired"]
        .mean()
        .reindex(season_order)
        .dropna()
        .reset_index()
    )
    season_chart = px.bar(
        season_summary,
        x="season_name",
        y="bikes_hired",
        color="bikes_hired",
        color_continuous_scale=["#17263a", "#22d3ee", "#2dd4bf"],
        title="Seasonal demand profile",
        labels={"season_name": "", "bikes_hired": "Average daily hires"},
    )
    season_chart.update_layout(coloraxis_showscale=False)
    season_chart.add_hline(
        y=average,
        line_dash="dot",
        line_color="#9aabc0",
        annotation_text="Period average",
    )

    driver_rows = []
    for variable in MODEL_VARIABLES:
        low, high = selected[variable].quantile([0.1, 0.9])
        driver_rows.append(
            {
                "Driver": VARIABLE_LABELS[variable].split(" (")[0],
                "Demand swing": COEFFICIENTS[variable] * (high - low),
            }
        )
    drivers = pd.DataFrame(driver_rows).sort_values("Demand swing")
    driver_chart = px.bar(
        drivers,
        x="Demand swing",
        y="Driver",
        orientation="h",
        color="Demand swing",
        color_continuous_scale=["#fb7185", "#2a384a", "#a3e635"],
        color_continuous_midpoint=0,
        title="Modelled weather leverage",
        labels={"Demand swing": "P10 → P90 impact"},
    )
    driver_chart.update_layout(coloraxis_showscale=False)

    best_day = (
        selected.groupby("day_of_week", observed=True)["bikes_hired"].mean().idxmax()
    )
    best_season = season_summary.loc[
        season_summary["bikes_hired"].idxmax(), "season_name"
    ]
    main_driver = drivers.iloc[
        drivers["Demand swing"].abs().argmax()
    ]["Driver"].lower()
    insight = (
        f"{best_season} is the strongest season in this window, while {best_day} "
        f"carries the highest average daily demand. The model identifies {main_driver} "
        f"as the largest weather lever; weekend demand is {abs(weekend_gap):.0%} "
        f"{'below' if weekend_gap < 0 else 'above'} weekday levels."
    )
    return (
        f"{total / 1_000_000:.1f}M",
        f"{average:,.0f}",
        f"{peak_row['bikes_hired']:,.0f}",
        peak_row["date"].strftime("%d %b %Y"),
        f"{weekend_gap:+.1%}",
        style_figure(trend, 360),
        style_figure(heatmap, 390),
        style_figure(season_chart, 390),
        style_figure(driver_chart, 390),
        insight,
    )


@callback(
    Output("weather-scatter", "figure"),
    Output("weekday-bar", "figure"),
    Input("explore-variable", "value"),
    Input("explore-colour", "value"),
)
def update_exploration(variable, colour):
    scatter = px.scatter(
        BIKES,
        x=variable,
        y="bikes_hired",
        color=colour,
        opacity=0.55,
        trendline="lowess",
        color_discrete_sequence=["#2dd4bf", "#22d3ee", "#fbbf24", "#fb7185"],
        labels={
            variable: VARIABLE_LABELS.get(variable, variable),
            "bikes_hired": "Daily bike hires",
        },
        title=f"Demand vs {VARIABLE_LABELS.get(variable, variable)}",
    )
    weekday = (
        BIKES.groupby("day_of_week", observed=True)["bikes_hired"]
        .mean()
        .reindex(DAY_ORDER)
        .reset_index()
    )
    bar = px.bar(
        weekday,
        x="day_of_week",
        y="bikes_hired",
        color="bikes_hired",
        color_continuous_scale=["#17263a", "#22d3ee", "#2dd4bf"],
        labels={"day_of_week": "", "bikes_hired": "Average daily hires"},
        title="Average demand by day of week",
    )
    bar.update_layout(coloraxis_showscale=False)
    return style_figure(scatter), style_figure(bar)


@callback(
    Output("model-time-series", "figure"),
    Output("residual-time-series", "figure"),
    Output("residual-distribution", "figure"),
    Input("model-date-range", "start_date"),
    Input("model-date-range", "end_date"),
    Input("model-frequency", "value"),
)
def update_model_series(start_date, end_date, frequency):
    selected = BIKES.loc[
        BIKES["date"].between(
            pd.Timestamp(start_date, tz="UTC"), pd.Timestamp(end_date, tz="UTC")
        ),
        ["date", "bikes_hired", "model_prediction", "residual"],
    ].copy()
    if selected.empty:
        empty = blank_figure("No observations in this date range")
        return empty, empty, empty

    series = (
        selected.set_index("date")[["bikes_hired", "model_prediction"]]
        .resample(frequency)
        .mean()
        .reset_index()
    )
    series_long = series.melt(
        "date",
        var_name="Series",
        value_name="Daily hires",
    )
    series_long["Series"] = series_long["Series"].map(
        {"bikes_hired": "Actual", "model_prediction": "Model L1"}
    )
    line = px.line(
        series_long,
        x="date",
        y="Daily hires",
        color="Series",
        title="Actual demand vs Model L1 prediction",
        color_discrete_map={"Actual": "#22d3ee", "Model L1": "#2dd4bf"},
    )

    residual_line = px.scatter(
        selected,
        x="date",
        y="residual",
        opacity=0.5,
        title="Residuals through time",
        labels={"residual": "Actual − predicted", "date": ""},
    )
    residual_line.add_hline(y=0, line_dash="dash", line_color="#fb7185")
    histogram = px.histogram(
        selected,
        x="residual",
        nbins=45,
        title="Residual distribution",
        labels={"residual": "Actual − predicted"},
        color_discrete_sequence=["#22d3ee"],
    )
    return (
        style_figure(line, 470),
        style_figure(residual_line),
        style_figure(histogram),
    )


@callback(
    Output("scenario-gauge", "figure"),
    Output("contribution-chart", "figure"),
    Output("scenario-summary", "children"),
    Input("scenario-temp", "value"),
    Input("scenario-precip", "value"),
    Input("scenario-windspeed", "value"),
    Input("scenario-visibility", "value"),
    Input("scenario-uvindex", "value"),
    Input("scenario-date", "date"),
)
def update_scenario(temp, precip, windspeed, visibility, uvindex, date):
    date_value = pd.Timestamp(date)
    day = date_value.strftime("%a")
    month = date_value.month
    values = {
        "temp": temp,
        "precip": precip,
        "windspeed": windspeed,
        "visibility": visibility,
        "uvindex": uvindex,
    }
    contributions = {
        VARIABLE_LABELS[name]: COEFFICIENTS[name] * value
        for name, value in values.items()
    }
    contributions["Day effect"] = COEFFICIENTS.get(f"day_{day}", 0.0)
    contributions["Month effect"] = COEFFICIENTS.get(f"month_{month}", 0.0)
    predicted = max(
        0,
        COEFFICIENTS["Intercept"] + sum(contributions.values()),
    )
    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=predicted,
            number={"valueformat": ",.0f"},
            title={"text": "Predicted daily hires"},
            gauge={
                "axis": {"range": [0, 80000]},
                "bar": {"color": "#2dd4bf"},
                "steps": [
                    {"range": [0, 18000], "color": "#3b1d28"},
                    {"range": [18000, 40000], "color": "#3a3018"},
                    {"range": [40000, 80000], "color": "#1d351e"},
                ],
            },
        )
    )
    contribution_frame = pd.DataFrame(
        {"Factor": contributions.keys(), "Contribution": contributions.values()}
    ).sort_values("Contribution")
    contribution = px.bar(
        contribution_frame,
        x="Contribution",
        y="Factor",
        orientation="h",
        color="Contribution",
        color_continuous_scale=["#fb7185", "#2a384a", "#a3e635"],
        color_continuous_midpoint=0,
        title="Contribution to prediction",
    )
    contribution.update_layout(coloraxis_showscale=False)
    historical_average = BIKES["bikes_hired"].mean()
    variance = predicted / historical_average - 1
    demand_band = (
        "Peak pressure"
        if predicted >= BIKES["bikes_hired"].quantile(0.75)
        else "Low utilization"
        if predicted <= BIKES["bikes_hired"].quantile(0.25)
        else "Normal operations"
    )
    strongest = max(contributions, key=lambda name: abs(contributions[name]))
    summary = [
        html.Div(
            [
                html.Div("VS NETWORK AVERAGE", className="decision-label"),
                html.Div(f"{variance:+.1%}", className="decision-value"),
                html.Div("expected demand variance", className="decision-note"),
            ],
            className="decision-card",
        ),
        html.Div(
            [
                html.Div("OPERATING SIGNAL", className="decision-label"),
                html.Div(demand_band, className="decision-value small"),
                html.Div(f"{day} · {date_value.strftime('%b')}", className="decision-note"),
            ],
            className="decision-card",
        ),
        html.Div(
            [
                html.Div("DOMINANT LEVER", className="decision-label"),
                html.Div(strongest, className="decision-value small"),
                html.Div(
                    f"{contributions[strongest]:+,.0f} hires",
                    className="decision-note",
                ),
            ],
            className="decision-card",
        ),
    ]
    return style_figure(gauge), style_figure(contribution), summary


def prediction_outputs(frame, title):
    frame = frame.copy()
    frame["predicted_hires"] = predict_hires(frame).round().astype(int)
    figure = px.bar(
        frame,
        x="date",
        y="predicted_hires",
        color="predicted_hires",
        color_continuous_scale=["#17263a", "#22d3ee", "#2dd4bf"],
        title=title,
        labels={"date": "", "predicted_hires": "Predicted hires"},
    )
    figure.update_layout(coloraxis_showscale=False)
    table = frame[
        ["date", "day_of_week", *MODEL_VARIABLES, "predicted_hires"]
    ].copy()
    table["date"] = pd.to_datetime(table["date"]).dt.strftime("%Y-%m-%d")
    for column in MODEL_VARIABLES:
        table[column] = table[column].round(1)
    return style_figure(figure, 350), table.to_dict("records")


@callback(
    Output("weather-status", "children"),
    Output("history-prediction", "figure"),
    Output("history-table", "data"),
    Output("history-table", "columns"),
    Output("forecast-prediction", "figure"),
    Output("forecast-table", "data"),
    Output("forecast-table", "columns"),
    Input("refresh-weather", "n_clicks"),
    Input("forecast-location", "value"),
)
def update_weather_predictions(_clicks, location):
    columns = [
        {"name": column.replace("_", " ").title(), "id": column}
        for column in ["date", "day_of_week", *MODEL_VARIABLES, "predicted_hires"]
    ]
    requested_location = location or "London"
    history_dates = pd.date_range("2026-01-01", "2026-01-07")
    forecast_dates = pd.date_range(pd.Timestamp.now().normalize(), periods=5)

    history, history_error = usable_weather(
        lambda: open_meteo_history(
            requested_location, "2026-01-01", "2026-01-07"
        ),
        history_dates,
    )
    forecast, forecast_error = usable_weather(
        lambda: open_meteo(requested_location, 5),
        forecast_dates,
    )
    history_figure, history_rows = prediction_outputs(
        history, "Historical-weather prediction"
    )
    forecast_figure, forecast_rows = prediction_outputs(
        forecast, "Five-day demand outlook"
    )

    def source_status(name, frame, error):
        source = frame.attrs.get("source", "weather data")
        if "cache_age_hours" in frame.attrs:
            source += f" ({frame.attrs['cache_age_hours']:.1f}h old)"
        if error:
            return (
                f"{name}: Open-Meteo unavailable ({error}); using {source}. "
                "These are model estimates, not observed/live weather."
            )
        note = f" {frame.attrs['note']}" if frame.attrs.get("note") else ""
        return f"{name}: {source}.{note}"

    status = " ".join(
        [
            source_status("Jan 1–7", history, history_error),
            source_status("Next five days", forecast, forecast_error),
        ]
    )
    return (
        status,
        history_figure,
        history_rows,
        columns,
        forecast_figure,
        forecast_rows,
        columns,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8050")), debug=False)
