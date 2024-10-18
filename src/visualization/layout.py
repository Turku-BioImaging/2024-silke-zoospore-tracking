import dash_bootstrap_components as dbc
from dash import html, dcc


def create_layout(replicates: list, metrics=list[dict]) -> dbc.Container:
    return dbc.Container(
        [
            dbc.Row(dbc.Col(html.H1("Zoospore data"), className="mb-4")),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4("Replicates"),
                            dcc.Checklist(
                                id="replicates-checklist",
                                options=[
                                    {"label": replicate, "value": replicate}
                                    for replicate in replicates
                                ],
                                value=replicates,
                                labelStyle={"display": "block"},
                            ),
                            html.H4("Metrics", className="mt-4"),
                            dcc.RadioItems(
                                id="metrics-radioitems",
                                options=metrics,
                                value=metrics[0]["value"],
                                labelStyle={"display": "block"},
                            ),
                        ],
                        width=2,
                    ),
                    dbc.Col(dcc.Graph(id="particle-tracking-graph"), width=10),
                ]
            ),
        ],
        fluid=True,
    )
