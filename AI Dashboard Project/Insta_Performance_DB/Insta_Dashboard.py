# app.py
# Plotly Dash app: Dark-themed 3x2 dashboard with cross-filtering from
# - Bar chart (content_category) selection
# - Heatmap (day_of_week + post_hour) click/selection
#
# Run:
#   pip install dash plotly pandas numpy
#   python app.py
#
# Notes:
# - This app loads Instagram_Analytics.csv from the same folder as app.py
# - Cross-filtering is done server-side in callbacks (true interactivity)
# - The bar supports click-to-filter (and box/lasso selection).
# - The heatmap supports click-to-filter (day and hour); click again / use Reset to clear.

import pandas as pd
import numpy as np
from datetime import datetime

from dash import Dash, dcc, html, Input, Output, State, callback_context, no_update
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ----------------------------
# Data loading + light cleaning
# ----------------------------
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="ascii")

    # Parse dates
    if "post_datetime" in df.columns:
        df["post_datetime"] = pd.to_datetime(df["post_datetime"], dayfirst=True, errors="coerce")
    if "post_date" in df.columns:
        df["post_date"] = pd.to_datetime(df["post_date"], dayfirst=True, errors="coerce")

    # Force numeric columns if present
    num_cols = [
        "post_hour", "likes", "comments", "shares", "saves", "reach", "impressions",
        "engagement_rate", "followers_gained", "caption_length", "hashtags_count",
        "has_call_to_action", "follower_count",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Categorical columns
    cat_cols = ["media_type", "content_category", "day_of_week"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("string")

    # Standardize day_of_week ordering
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    if "day_of_week" in df.columns:
        df["day_of_week"] = pd.Categorical(df["day_of_week"], categories=dow_order, ordered=True)

    return df


DF = load_data("./Instagram_Analytics.csv")


# ----------------------------
# Filtering helpers
# ----------------------------
def apply_filters(
    df: pd.DataFrame,
    selected_categories: list | None,
    heat_filter: dict | None,
) -> pd.DataFrame:
    out_df = df

    if selected_categories:
        out_df = out_df[out_df["content_category"].isin(selected_categories)]

    if heat_filter and (heat_filter.get("day_of_week") is not None):
        out_df = out_df[out_df["day_of_week"].astype("string") == str(heat_filter["day_of_week"])]

    if heat_filter and (heat_filter.get("post_hour") is not None):
        out_df = out_df[out_df["post_hour"] == int(heat_filter["post_hour"])]

    return out_df


def parse_bar_selection(click_data, selected_data) -> list | None:
    # Prefer lasso/box selection if present; fall back to click
    if selected_data and "points" in selected_data and len(selected_data["points"]) > 0:
        cats = []
        for pt in selected_data["points"]:
            if "x" in pt and pt["x"] is not None:
                cats.append(str(pt["x"]))
        cats = sorted(list(set(cats)))
        return cats if len(cats) > 0 else None

    if click_data and "points" in click_data and len(click_data["points"]) > 0:
        x_val = click_data["points"][0].get("x", None)
        if x_val is not None:
            return [str(x_val)]

    return None


def parse_heat_interaction(click_data, selected_data) -> dict | None:
    # Heatmap: click gives one cell; selection can give multiple cells (we'll use the first for a clean filter)
    if selected_data and "points" in selected_data and len(selected_data["points"]) > 0:
        pt0 = selected_data["points"][0]
        day = pt0.get("y", None)
        hour = pt0.get("x", None)
        return {"day_of_week": day, "post_hour": hour}

    if click_data and "points" in click_data and len(click_data["points"]) > 0:
        pt0 = click_data["points"][0]
        day = pt0.get("y", None)
        hour = pt0.get("x", None)
        return {"day_of_week": day, "post_hour": hour}

    return None


# ----------------------------
# Figure builders (all Plotly Dark)
# ----------------------------
def fig_engagement_by_category(df: pd.DataFrame) -> go.Figure:
    agg = (
        df.groupby("content_category", dropna=False)["engagement_rate"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    fig = px.bar(
        agg,
        x="content_category",
        y="engagement_rate",
        template="plotly_dark",
        color_discrete_sequence=["#60a5fa"],
    )
    fig.update_traces(
        hovertemplate="Category=%{x}<br>Avg engagement rate=%{y:.4f}<extra></extra>"
    )
    fig.update_layout(
        title="Avg Engagement Rate by Content Category",
        margin=dict(l=40, r=20, t=55, b=40),
    )
    fig.update_xaxes(title="Content Category")
    fig.update_yaxes(title="Avg Engagement Rate", tickformat=".2%")
    return fig


def fig_media_type_pie(df: pd.DataFrame) -> go.Figure:
    counts = df["media_type"].value_counts(dropna=False).reset_index()
    counts.columns = ["media_type", "count"]
    fig = px.pie(
        counts,
        names="media_type",
        values="count",
        template="plotly_dark",
        color_discrete_sequence=["#f59e0b", "#60a5fa", "#a78bfa", "#34d399", "#fb7185"],
    )
    fig.update_traces(
        textinfo="percent",
        hovertemplate="Media type=%{label}<br>Count=%{value:,}<br>Share=%{percent}<extra></extra>",
    )
    fig.update_layout(
        title="Media Type Distribution",
        margin=dict(l=40, r=20, t=55, b=40),
        legend_title_text="",
    )
    return fig


def fig_reach_trend(df: pd.DataFrame) -> go.Figure:
    ts = (
        df.dropna(subset=["post_date"])
        .groupby("post_date")["reach"]
        .sum()
        .sort_index()
        .reset_index()
    )
    ts["reach_7d"] = ts["reach"].rolling(7, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ts["post_date"],
            y=ts["reach"],
            mode="lines",
            name="Daily reach",
            line=dict(color="#60a5fa", width=1),
            opacity=0.45,
            hovertemplate="Date=%{x|%Y-%m-%d}<br>Reach=%{y:,}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ts["post_date"],
            y=ts["reach_7d"],
            mode="lines",
            name="7-day rolling avg",
            line=dict(color="#fb7185", width=3),
            hovertemplate="Date=%{x|%Y-%m-%d}<br>7d avg reach=%{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        title="Reach Over Time",
        margin=dict(l=40, r=20, t=55, b=40),
        hovermode="x",
    )
    fig.update_xaxes(title="Post Date")
    fig.update_yaxes(title="Total Reach")
    return fig


def fig_likes_vs_saves(df: pd.DataFrame) -> go.Figure:
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    agg = (
        df.groupby("day_of_week")[["likes", "saves"]]
        .mean()
        .reindex(dow_order)
        .reset_index()
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=agg["day_of_week"].astype(str),
            y=agg["likes"],
            name="Avg likes",
            marker_color="#60a5fa",
            hovertemplate="Day=%{x}<br>Avg likes=%{y:,.1f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=agg["day_of_week"].astype(str),
            y=agg["saves"],
            name="Avg saves",
            marker_color="#34d399",
            hovertemplate="Day=%{x}<br>Avg saves=%{y:,.1f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        title="Avg Likes vs Saves by Day of Week",
        margin=dict(l=40, r=20, t=55, b=40),
        barmode="group",
    )
    fig.update_xaxes(title="Day of Week")
    fig.update_yaxes(title="Average")
    return fig


def fig_reach_vs_impressions(df: pd.DataFrame) -> go.Figure:
    scat = df[["impressions", "reach", "media_type", "content_category", "day_of_week"]].dropna()
    if len(scat) > 25000:
        scat = scat.sample(25000, random_state=42)

    # Color by media_type to keep legends meaningful + clickable
    fig = px.scatter(
        scat,
        x="impressions",
        y="reach",
        color=scat["media_type"].astype(str),
        template="plotly_dark",
        opacity=0.35,
        color_discrete_map={
            "reel": "#f59e0b",
            "image": "#60a5fa",
            "carousel": "#a78bfa",
        },
    )
    fig.update_traces(
        marker=dict(size=7),
        hovertemplate=(
            "Impressions=%{x:,}<br>"
            "Reach=%{y:,}<br>"
            "Media=%{legendgroup}<extra></extra>"
        ),
    )
    fig.update_layout(
        title="Reach vs Impressions",
        margin=dict(l=40, r=20, t=55, b=40),
    )
    fig.update_xaxes(title="Impressions")
    fig.update_yaxes(title="Reach")

    # Selection styling
    fig.update_traces(selected=dict(marker=dict(opacity=0.9, size=10)))
    fig.update_traces(unselected=dict(marker=dict(opacity=0.12)))
    return fig


def fig_engagement_heatmap(df: pd.DataFrame) -> go.Figure:
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heat = df.pivot_table(
        index="day_of_week",
        columns="post_hour",
        values="engagement_rate",
        aggfunc="mean",
    ).reindex(dow_order)

    x_hours = [int(x) for x in heat.columns]
    y_days = [str(y) for y in heat.index]

    fig = go.Figure(
        data=go.Heatmap(
            z=heat.values,
            x=x_hours,
            y=y_days,
            colorscale="Viridis",
            colorbar=dict(title="Avg ER"),
            hovertemplate="Day=%{y}<br>Hour=%{x}<br>Avg ER=%{z:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        title="Avg Engagement Rate Heatmap (Day x Hour)",
        margin=dict(l=40, r=20, t=55, b=40),
    )
    fig.update_xaxes(title="Post Hour")
    fig.update_yaxes(title="Day of Week")
    return fig


# ----------------------------
# Dash app layout
# ----------------------------
app = Dash(__name__)
server = app.server

app.layout = html.Div(
    style={
        "backgroundColor": "#0b1220",
        "minHeight": "100vh",
        "padding": "14px 16px 20px 16px",
        "fontFamily": "system-ui, -apple-system, Segoe UI, Roboto, Arial",
    },
    children=[
        html.Div(
            style={"maxWidth": "1400px", "margin": "0 auto"},
            children=[
                html.Div(
                    style={
                        "display": "flex",
                        "justifyContent": "space-between",
                        "alignItems": "center",
                        "gap": "12px",
                        "marginBottom": "10px",
                    },
                    children=[
                        html.Div(
                            children=[
                                html.Div(
                                    "Instagram Performance Executive Dashboard",
                                    style={
                                        "color": "#e5e7eb",
                                        "fontSize": "22px",
                                        "fontWeight": "700",
                                        "lineHeight": "1.1",
                                    },
                                ),
                                html.Div(
                                    "Cross-filter from category bar or heatmap cells (day/hour).",
                                    style={"color": "#9ca3af", "marginTop": "6px", "fontSize": "13px"},
                                ),
                            ]
                        ),
                        html.Button(
                            "Reset filters",
                            id="btn-reset",
                            n_clicks=0,
                            style={
                                "background": "#111827",
                                "border": "1px solid #22304f",
                                "color": "#e5e7eb",
                                "padding": "8px 12px",
                                "borderRadius": "10px",
                                "cursor": "pointer",
                            },
                        ),
                    ],
                ),

                # Stores for filters
                dcc.Store(id="store-category-filter", data=None),
                dcc.Store(id="store-heat-filter", data=None),

                # Status line
                html.Div(
                    id="filter-status",
                    style={"color": "#cbd5e1", "marginBottom": "10px", "fontSize": "13px"},
                ),

                # Responsive grid
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(2, minmax(0, 1fr))",
                        "gap": "12px",
                    },
                    children=[
                        dcc.Graph(
                            id="g-bar-category",
                            figure=fig_engagement_by_category(DF),
                            config={"displaylogo": False, "responsive": True},
                            style={"height": "360px"},
                            clear_on_unhover=True,
                        ),
                        dcc.Graph(
                            id="g-pie-media",
                            figure=fig_media_type_pie(DF),
                            config={"displaylogo": False, "responsive": True},
                            style={"height": "360px"},
                        ),
                        dcc.Graph(
                            id="g-line-reach",
                            figure=fig_reach_trend(DF),
                            config={"displaylogo": False, "responsive": True},
                            style={"height": "360px"},
                        ),
                        dcc.Graph(
                            id="g-bar-likes-saves",
                            figure=fig_likes_vs_saves(DF),
                            config={"displaylogo": False, "responsive": True},
                            style={"height": "360px"},
                        ),
                        dcc.Graph(
                            id="g-scatter",
                            figure=fig_reach_vs_impressions(DF),
                            config={"displaylogo": False, "responsive": True},
                            style={"height": "360px"},
                        ),
                        dcc.Graph(
                            id="g-heatmap",
                            figure=fig_engagement_heatmap(DF),
                            config={"displaylogo": False, "responsive": True},
                            style={"height": "360px"},
                        ),
                    ],
                ),

                html.Div(
                    style={"color": "#9ca3af", "marginTop": "10px", "fontSize": "12px"},
                    children=[
                        "Tip: Use box/lasso select on the category bar to select multiple categories. ",
                        "Click a heatmap cell to filter by that day/hour.",
                    ],
                ),
            ],
        )
    ],
)


# ----------------------------
# Callbacks: update filter stores
# ----------------------------
@app.callback(
    Output("store-category-filter", "data"),
    Output("store-heat-filter", "data"),
    Input("g-bar-category", "clickData"),
    Input("g-bar-category", "selectedData"),
    Input("g-heatmap", "clickData"),
    Input("g-heatmap", "selectedData"),
    Input("btn-reset", "n_clicks"),
    State("store-category-filter", "data"),
    State("store-heat-filter", "data"),
)
def update_filters(bar_click, bar_selected, heat_click, heat_selected, reset_clicks, cat_state, heat_state):
    trig = callback_context.triggered[0]["prop_id"] if callback_context.triggered else ""

    # Reset
    if trig == "btn-reset.n_clicks":
        return None, None

    # Start from existing state (so filters can stack)
    new_cat = cat_state
    new_heat = heat_state

    # Category filter updated by bar interactions
    if trig in ("g-bar-category.clickData", "g-bar-category.selectedData"):
        parsed = parse_bar_selection(bar_click, bar_selected)
        new_cat = parsed

    # Heat filter updated by heat interactions
    if trig in ("g-heatmap.clickData", "g-heatmap.selectedData"):
        parsed = parse_heat_interaction(heat_click, heat_selected)
        new_heat = parsed

    return new_cat, new_heat


# ----------------------------
# Callback: redraw all charts based on filters
# ----------------------------
@app.callback(
    Output("g-bar-category", "figure"),
    Output("g-pie-media", "figure"),
    Output("g-line-reach", "figure"),
    Output("g-bar-likes-saves", "figure"),
    Output("g-scatter", "figure"),
    Output("g-heatmap", "figure"),
    Output("filter-status", "children"),
    Input("store-category-filter", "data"),
    Input("store-heat-filter", "data"),
)
def redraw_dashboard(selected_categories, heat_filter):
    fdf = apply_filters(DF, selected_categories, heat_filter)

    # Rebuild figs from filtered data
    f1 = fig_engagement_by_category(fdf)
    f2 = fig_media_type_pie(fdf)
    f3 = fig_reach_trend(fdf)
    f4 = fig_likes_vs_saves(fdf)
    f5 = fig_reach_vs_impressions(fdf)
    f6 = fig_engagement_heatmap(fdf)

    # Status text (human readable)
    parts = []
    if selected_categories:
        if len(selected_categories) == 1:
            parts.append("Category filter: " + str(selected_categories[0]))
        else:
            parts.append("Category filter: " + str(len(selected_categories)) + " categories")
    if heat_filter and (heat_filter.get("day_of_week") is not None or heat_filter.get("post_hour") is not None):
        day = heat_filter.get("day_of_week", None)
        hour = heat_filter.get("post_hour", None)
        if day is not None and hour is not None:
            parts.append("Time filter: " + str(day) + " @ " + str(int(hour)) + ":00")
        elif day is not None:
            parts.append("Day filter: " + str(day))
        elif hour is not None:
            parts.append("Hour filter: " + str(int(hour)) + ":00")

    if len(parts) == 0:
        status = "Filters: none (showing all posts)"
    else:
        status = " | ".join(parts) + " (showing " + str(len(fdf)) + " posts)"

    return f1, f2, f3, f4, f5, f6, status


if __name__ == "__main__":
    app.run_server(debug=True, host="127.0.0.1", port=8050)
