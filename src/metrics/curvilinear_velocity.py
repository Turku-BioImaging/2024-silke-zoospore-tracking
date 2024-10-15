import os

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from tqdm import tqdm

import constants

TRACKING_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "tracking_data"
)
TRACKING_FNAME = "tracking_derived.csv"
PIXEL_SIZE = constants.PIXEL_SIZE
FRAME_INTERVAL_REGULAR = constants.FRAME_INTERVAL_REGULAR
FRAME_INTERVAL_LOW_LIGHT = constants.FRAME_INTERVAL_LOW_LIGHT


def _get_particle_data(particle_id: int, df: pl.DataFrame) -> dict:
    # particle id
    # get all frames where the particle is present
    # sum the values from the column "displacement_(um)"
    # calculate the total time itnerval in seconds between first and last frames

    particle_data = df.filter(pl.col("particle") == particle_id)

    displacement = particle_data.select(pl.col("displacement_(um)")).sum().item()

    frame_interval = df.select(pl.col("frame_interval"))[0].item()
    first_frame = particle_data.select(pl.col("frame")).min()
    last_frame = particle_data.select(pl.col("frame")).max()
    total_time = (last_frame.item() - first_frame.item()) * frame_interval
    curvilinear_velocity = (displacement / total_time) if total_time != 0 else np.nan

    return {
        "particle_id": particle_id,
        "total_displacement_(um)": displacement,
        "total_time_(s)": total_time,
        "curvilinear_velocity_(um/s)": curvilinear_velocity,
    }


def process_all_data():
    tracking_data = [
        (replicate, sample)
        for replicate in os.listdir(TRACKING_DATA_DIR)
        if os.path.isdir(os.path.join(TRACKING_DATA_DIR, replicate))
        for sample in os.listdir(os.path.join(TRACKING_DATA_DIR, replicate))
        if os.path.isdir(os.path.join(TRACKING_DATA_DIR, replicate, sample))
    ]

    def process_tracking_data(td):
        csv_path = os.path.join(TRACKING_DATA_DIR, td[0], td[1], TRACKING_FNAME)
        df = pl.read_csv(csv_path)
        particle_ids = (
            df.select(pl.col("particle"))
            .unique()
            .sort("particle")
            .to_series()
            .to_list()
        )

        particle_cl_velocities = []

        for particle_id in particle_ids:
            particle_cl_velocities.append(_get_particle_data(particle_id, df))

        particle_cl_velocities_df = pl.DataFrame(particle_cl_velocities)
        particle_cl_velocities_df.write_csv(
            os.path.join(TRACKING_DATA_DIR, td[0], td[1], "curvilinear_velocities.csv")
        )

    Parallel(n_jobs=-1)(
        delayed(process_tracking_data)(td)
        for td in tqdm(tracking_data, desc="Calculating curvilinear velocities")
    )


if __name__ == "__main__":
    process_all_data()
