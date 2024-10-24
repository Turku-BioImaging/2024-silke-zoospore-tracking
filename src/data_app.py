"""
Run this script to start the Dash app that displays the particle tracking data.
"""

from dash import Dash, Input, Output, callback_context
import plotly.express as px
import polars as pl
import dash_bootstrap_components as dbc
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import random
from visualization.layout import create_layout
from dotenv import load_dotenv

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "tracking_data")
METRICS = [
    {
        "label": "Average speed (um/s)",
        "value": "average_speed_(um/s)",
    },
    {
        "label": "Curvilinear velocity (um/s)",
        "value": "curvilinear_velocity_(um/s)",
    },
    {
        "label": "Straight-line velocity (um/s)",
        "value": "straight_line_velocity_(um/s)",
    },
    {
        "label": "Directionality ratio",
        "value": "directionality_ratio",
    },
    {
        "label": "Area covered (um^2)",
        "value": "area_covered_(um^2)",
    },
    {
        "label": "Direction change frequency (Hz)",
        "value": "direction_change_frequency_(Hz)",
    },
]


def load_particle_data() -> pl.DataFrame:
    sample_data = [
        (replicate, sample)
        for replicate in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, replicate))
        for sample in os.listdir(os.path.join(DATA_DIR, replicate))
        if os.path.isdir(os.path.join(DATA_DIR, replicate, sample))
    ]

    random.shuffle(sample_data)

    def compile_particle_data(td):
        replicate, sample = td
        tracks_df = pl.read_csv(os.path.join(DATA_DIR, replicate, sample, "tracks.csv"))
        particle_df = pl.read_csv(
            os.path.join(DATA_DIR, replicate, sample, "particles.csv")
        )

        test = tracks_df["test"][0]
        step_init_abs = str(tracks_df["step_init_abs"][0]).zfill(2)
        step_end_abs = str(tracks_df["step_end_abs"][0]).zfill(2)

        data_dict = {
            "replicate": replicate,
            "sample": sample,
            "test": test,
            # "step_init_abs": step_init_abs,
            # "step_end_abs": step_end_abs,
            "step": f"{step_init_abs}-{step_end_abs}",
        }

        return pl.DataFrame({**data_dict, **particle_df.to_dict()})

    all_particle_df = Parallel(n_jobs=-1)(
        delayed(compile_particle_data)(td)
        for td in tqdm(sample_data, desc="Loading particle data")
    )

    all_particle_df = pl.concat(all_particle_df)
    all_particle_df = all_particle_df.sort(
        [
            "replicate",
            "sample",
            "particle_id",
        ]
    )


    return all_particle_df


particles_df = load_particle_data()

# Load environment variables from a .env file
# load_dotenv()

def _get_url_base_pathname():
    load_dotenv(dotenv_path=".env")
    load_dotenv(dotenv_path=".env.local")
    
    if os.getenv('URL_BASE_PATHNAME'):
        return os.getenv('URL_BASE_PATHNAME')
    else:
        return "/"

app = Dash(__name__, external_stylesheets=[dbc.themes.SLATE], url_base_pathname=_get_url_base_pathname())
app.layout = create_layout(
    replicates=sorted(particles_df["replicate"].unique().to_list()),
    metrics=METRICS,
    steps=sorted(particles_df["step"].unique(), reverse=True),
)


@app.callback(
    Output("particle-tracking-graph", "figure"),
    Input("replicates-checklist", "value"),
    Input("metrics-radioitems", "value"),
    Input("steps-dropdown", "value"),
)
def update_graph(selected_replicates: list, selected_metric: str, selected_steps: list):
    filtered_particles_df = particles_df.filter(
        (pl.col("replicate").is_in(selected_replicates))
        & (pl.col("step").is_in(selected_steps))
    )

    # print(selected_metric)
    filtered_metric = list(filter(lambda x: x["value"] == selected_metric, METRICS))

    fig = px.strip(
        filtered_particles_df,
        y=selected_metric,
        color="replicate",
        facet_col="step",
        facet_col_wrap=10,
        height=1024,
        labels={selected_metric: filtered_metric[0]["label"]},
        template="plotly_dark",
        category_orders={
            "step": sorted(filtered_particles_df["step"].unique(), reverse=True)
        },
    )

    return fig


@app.callback(
    Output("steps-dropdown", "value"),
    Input("select-all-button", "n_clicks"),
    Input("select-none-button", "n_clicks"),
)
def update_steps_dropdown(select_all_clicks, select_none_clicks):
    ctx = callback_context

    if not ctx.triggered:
        return sorted(particles_df["step"].unique(), reverse=True)

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "select-all-button":
        return sorted(particles_df["step"].unique(), reverse=True)
    elif button_id == "select-none-button":
        return []

    return sorted(particles_df["step"].unique(), reverse=True)


if __name__ == "__main__":
    app.run_server(debug=True)
