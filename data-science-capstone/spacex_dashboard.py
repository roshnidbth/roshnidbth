"""Interactive Plotly Dash view of launch-site and payload outcomes."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, dcc, html


DATA_PATH = "spacex_launch_dash.csv"
df = pd.read_csv(DATA_PATH)
payload_min = float(df["Payload Mass (kg)"].min())
payload_max = float(df["Payload Mass (kg)"].max())

app = Dash(__name__)
app.layout = html.Div(
    [
        html.H1("SpaceX Launch Records Dashboard"),
        dcc.Dropdown(
            id="site-dropdown",
            options=[{"label": "All Sites", "value": "ALL"}]
            + [{"label": site, "value": site} for site in sorted(df["Launch Site"].unique())],
            value="ALL",
            clearable=False,
        ),
        dcc.RangeSlider(id="payload-slider", min=payload_min, max=payload_max,
                        value=[payload_min, payload_max]),
        dcc.Graph(id="success-pie"),
        dcc.Graph(id="payload-scatter"),
    ]
)


@app.callback(
    Output("success-pie", "figure"),
    Output("payload-scatter", "figure"),
    Input("site-dropdown", "value"),
    Input("payload-slider", "value"),
)
def update_dashboard(site: str, payload_range: list[float]):
    filtered = df[df["Payload Mass (kg)"].between(*payload_range)]
    if site != "ALL":
        filtered = filtered[filtered["Launch Site"] == site]
    pie = px.pie(filtered, names="class", title="Launch outcome share")
    scatter = px.scatter(filtered, x="Payload Mass (kg)", y="class", color="Booster Version Category",
                         hover_data=["Launch Site"], title="Payload mass vs. launch outcome")
    return pie, scatter


if __name__ == "__main__":
    app.run(debug=True)
