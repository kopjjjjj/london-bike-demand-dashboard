# London Bike Intelligence

A fintech-style business intelligence dashboard for exploring and forecasting
daily London bike demand. The application is built with Dash and uses our final
OLS regression model (`Model L1`) together with Open-Meteo weather data.

## Links

- **Live dashboard:** https://london-bike-demand.onrender.com
- **GitHub repository:** https://github.com/kopjjjjj/london-bike-demand-dashboard
- **London bikes dataset:** https://raw.githubusercontent.com/kostis-christodoulou/am01-code-sep2026/main/data/london_bikes.csv

> Render's free service may sleep when inactive. The first visit can take
> approximately 30–60 seconds to load.

## Dashboard features

- Executive overview with KPI cards, date and season filters, demand trends,
  calendar heatmap, and modelled weather drivers
- Interactive exploration of bike demand against weather variables
- Model Lab with weekly, monthly, quarterly, and yearly time-series views
- Actual-versus-predicted demand and residual diagnostics
- Scenario Studio with manual weather inputs, a demand gauge, and coefficient
  contribution analysis
- Historical predictions for 1–7 January 2026 and a live five-day forecast
- Filterable and sortable raw-data workbench

## Final model

Model L1 is:

```text
bikes_hired ~ C(day_of_week) + temp + precip + windspeed
              + C(month) + visibility + uvindex
```

Model performance on the training data:

- Adjusted R-squared: `0.600`
- Residual standard error: `5,812` daily hires
- All final numeric-predictor VIF values are below `1.6`

The coefficients used by the dashboard are stored in
`model_coefficients.csv`. Historical Open-Meteo records do not always include
visibility or UV index, so the app clearly uses training-data climatology
fallbacks when those fields are unavailable.

## Project files

```text
.
├── app.py                    # Dash application and callbacks
├── open_meteo.py             # Forecast and historical weather helpers
├── model_coefficients.csv    # Exported Model L1 coefficients
├── bikes_assignment.ipynb    # Completed analysis notebook
├── pyproject.toml            # Python project and dependencies
├── uv.lock                   # Reproducible dependency lock
├── render.yaml               # Render deployment configuration
└── README.md                 # Project documentation
```

## Build and run locally

### Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)

Clone the repository:

```bash
git clone https://github.com/kopjjjjj/london-bike-demand-dashboard.git
cd london-bike-demand-dashboard
```

Install the locked dependencies:

```bash
uv sync
```

Start the dashboard:

```bash
uv run python app.py
```

Open http://127.0.0.1:8050 in a browser.

To check the production server configuration locally:

```bash
uv run gunicorn --check-config app:server
```

## Upload a new project to GitHub

If these files are in a new local folder, create and publish a public GitHub
repository with:

```bash
git init
git add app.py open_meteo.py model_coefficients.csv pyproject.toml uv.lock render.yaml README.md
git commit -m "Build London bike demand dashboard"
git branch -M main
gh repo create london-bike-demand-dashboard --public --source=. --remote=origin --push
```

This workflow requires the
[GitHub CLI](https://cli.github.com/) and an authenticated session:

```bash
gh auth login
```

For later updates:

```bash
git add .
git commit -m "Update dashboard"
git push origin main
```

## Deploy on Render

### One-click deployment

Open:

https://render.com/deploy?repo=https://github.com/kopjjjjj/london-bike-demand-dashboard

The included `render.yaml` supplies the service configuration.

### Manual deployment

1. Sign in to [Render](https://render.com/).
2. Select **New > Web Service**.
3. Connect `kopjjjjj/london-bike-demand-dashboard`.
4. Use the `main` branch.
5. Set the build command to:

   ```bash
   pip install uv && uv sync
   ```

6. Set the start command to:

   ```bash
   uv run gunicorn app:server
   ```

7. Deploy and wait for the service status to become **Live**.

No API key or environment variable is required. The application exposes
`server = app.server` for Gunicorn and reads Render's `PORT` automatically.

## Updating model coefficients

After changing the final model in `bikes_assignment.ipynb`:

1. Run Parts 2–5 of the notebook.
2. Confirm that `model_coefficients.csv` was regenerated.
3. Ensure the variables in `MODEL_VARIABLES` inside `app.py` match the CSV.
4. Test locally with `uv run python app.py`.
5. Commit and push the updated files; Render will redeploy automatically.
