import json
import os
import re

import constants
import numpy as np
import duckdb
import pandas as pd
import polars as pl
import trackpy as tp
from joblib import Parallel, delayed
import sqlite3

TRACKING_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "tracking_data"
)

DATABASE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "database", "tracking.db"
)

PIXEL_SIZE = constants.PIXEL_SIZE
FRAME_INTERVAL_REGULAR = constants.FRAME_INTERVAL_REGULAR
FRAME_INTERVAL_LOW_LIGHT = constants.FRAME_INTERVAL_LOW_LIGHT

with open(
    os.path.join(os.path.dirname(__file__), "light_intensity_codes.json"), "r"
) as f:
    light_intensity_codes = json.load(f)


def classify_sample(replicate: str, sample: str, light_intensity_codes: dict) -> dict:
    # Get code of init light level
    index = sample.find("_from")
    assert index != -1, f"Invalid sample name: {sample}"
    step_init = int(sample[index + 5])
    step_init_abs = light_intensity_codes[str(step_init)]

    # Get code of end light level
    step_end = int(sample[index - 1])
    step_end_abs = light_intensity_codes[str(step_end)]

    # get step type
    if step_init_abs == step_end_abs:
        step_type = "adaptation"
    if step_init_abs < step_end_abs:
        step_type = "step_up"
    if step_init_abs > step_end_abs:
        step_type = "step_down"

    # Get species
    if "AstChy1" in replicate:
        species = "AstChy1"
    elif "StauChy1" in replicate:
        species = "StauChy1"
    else:
        raise ValueError(f"Could not infer species name: {replicate}/{sample}")

    # get replicate
    index = replicate.find("rep")
    replicate_number = int(replicate[index + 3])

    # get test number
    match = re.search("test\d{1,2}", sample)
    if match:
        digits = match.group()[4:]
        test = int(digits)
    else:
        raise ValueError(f"Could not infer test number: {replicate}/{sample}")

    # get frame interval
    if step_end == 5:
        # frame_interval = 0.11237
        frame_interval = FRAME_INTERVAL_LOW_LIGHT
    else:
        # frame_interval = 0.02729
        frame_interval = FRAME_INTERVAL_REGULAR

    return {
        "replicate": replicate,
        "sample": sample,
        "species": species,
        "replicate_number": replicate_number,
        "test": test,
        "step_init": step_init,
        "step_end": step_end,
        "step_init_abs": step_init_abs,
        "step_end_abs": step_end_abs,
        "step_type": step_type,
        "frame_interval": frame_interval,
    }


def calculate_speeds(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort(["particle", "frame"])

    df = df.with_columns(
        [
            pl.col("x").cast(pl.Float64),
            pl.col("y").cast(pl.Float64),
        ]
    )

    df = df.with_columns(
        [
            ((pl.col("x") - pl.col("x").shift(1)).over("particle") * PIXEL_SIZE).alias(
                "dx_um"
            ),
            ((pl.col("y") - pl.col("y").shift(1)).over("particle") * PIXEL_SIZE).alias(
                "dy_um"
            ),
        ]
    )

    df = df.with_columns(
        (pl.col("dx_um").pow(2) + pl.col("dy_um").pow(2))
        .sqrt()
        .alias("displacement_um"),
    )

    # df = df.with_columns(
    #     (pl.col("displacement_(um)") / pl.col("frame_interval")).alias("speed_(um/s)"),
    # )

    return df


def main():
    # conn = sqlite3.connect(DATABASE_PATH)

    with open(
        os.path.join(os.path.dirname(__file__), "light_intensity_codes.json"), "r"
    ) as f:
        light_intensity_codes = json.load(f)

    # init database
    conn = sqlite3.connect(DATABASE_PATH)

    cur = conn.cursor()
    with open(
        os.path.join(os.path.dirname(__file__), "sql", "create_tables.sql"), "r"
    ) as f:
        cur.executescript(f.read())

    conn.commit()
    conn.close()

    # gather replicate and sample data dirs
    tracks_data = [
        (replicate, sample)
        for replicate in os.listdir(TRACKING_DATA_DIR)
        if os.path.isdir(os.path.join(TRACKING_DATA_DIR, replicate))
        for sample in os.listdir(os.path.join(TRACKING_DATA_DIR, replicate))
        if os.path.isdir(os.path.join(TRACKING_DATA_DIR, replicate, sample))
    ]

    def process_tracks_data(replicate, sample):
        df = pl.read_csv(
            os.path.join(TRACKING_DATA_DIR, replicate, sample, "tracking.csv")
        )
        if "Unnamed: 0" in df.columns:
            df = df.drop("Unnamed: 0")

        sample_descriptors = classify_sample(replicate, sample, light_intensity_codes)

        for col_name, value in sample_descriptors.items():
            df = df.with_columns(pl.lit(value).alias(col_name))

        df = calculate_speeds(df)

        # # write imsd and emsd data
        # fps = 1 / (df["frame_interval"][0])
        # df_pandas = df.to_pandas()
        # im = tp.imsd(df_pandas, mpp=PIXEL_SIZE, fps=fps, max_lagtime=450)
        # im.to_csv(os.path.join(TRACKING_DATA_DIR, replicate, sample, "imsd.csv"))

        # em = tp.emsd(df_pandas, mpp=PIXEL_SIZE, fps=fps, max_lagtime=450)
        # em.to_csv(os.path.join(TRACKING_DATA_DIR, replicate, sample, "emsd.csv"))

        # write derived tracking data
        df = df.select(
            [
                "replicate",
                "sample",
                "frame",
                "particle",
                "x",
                "y",
                "test",
                "step_init",
                "step_end",
                "step_init_abs",
                "step_end_abs",
                "step_type",
                "frame_interval",
                "dx_um",
                "dy_um",
                "displacement_um",
            ]
        )

        df.write_database(
            "tracks", connection=f"sqlite:///{DATABASE_PATH}", if_table_exists="append"
        )

    for replicate, sample in tqdm(tracks_data):
        process_tracks_data(replicate, sample)


if __name__ == "__main__":
    process_all_tracks()
