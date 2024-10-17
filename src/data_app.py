from dash import Dash, dcc, html
import plotly.express as px
import polars as pl
import dash_bootstrap_components as dbc
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import random

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "tracking_data")


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
        ["replicate", "sample", "particle_id", "test"]
    )

    return all_particle_df


if __name__ == "__main__":
    particles_df = load_particle_data()

    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    fig = px.strip(
        particles_df.to_pandas(),
        y="straight_line_velocity_(um/s)",
        color="replicate",
        facet_col="step",
        facet_col_wrap=10,
        height=1024,
        labels={"straight_line_velocity_(um/s)": "SLV (µm/s)"},
        template="plotly_white",
    )

    app.layout = dbc.Container(
        [
            dbc.Row(dbc.Col(html.H1("Zoospore data"), className="mb-4")),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4("Select replicates to display:"),
                            dcc.Checklist(
                                id="replicates",
                                options=[
                                    {"label": replicate, "value": replicate}
                                    for replicate in particles_df["replicate"]
                                    .unique()
                                    .to_list()
                                ],
                                value=particles_df["replicate"].unique().to_list(),
                                labelStyle={"display": "block"},
                            ),
                        ],
                        width=2,
                    ),
                    dbc.Col(dcc.Graph(id="particle-tracking-graph", figure=fig), width=10),
                ]
            ),
        ],
        fluid=True,
    )
    app.run_server(debug=True)
