import os
import polars as pl
from joblib import Parallel, delayed
from tqdm import tqdm
import constants
import numpy as np

TRACKING_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "tracking_data"
)
PIXEL_SIZE = constants.PIXEL_SIZE
FRAME_INTERVAL_REGULAR = constants.FRAME_INTERVAL_REGULAR
FRAME_INTERVAL_LOW_LIGHT = constants.FRAME_INTERVAL_LOW_LIGHT


def _get_particle_data(particle_id: int, df: pl.DataFrame) -> dict:
    # particle id
    # Get first and last frames where the particle is present.
    # Get the total path length of the particle.
    # get the total time itnerval in seconds between first and last frames.
    particle_data = df.filter(pl.col("particle") == particle_id)

    frame_interval = df.select(pl.col("frame_interval"))[0].item()
    first_frame = particle_data.select(pl.col("frame")).min()
    last_frame = particle_data.select(pl.col("frame")).max()
    total_time = (last_frame.item() - first_frame.item()) * frame_interval

    displacement = particle_data.select(pl.col("displacement_(um)")).sum().item()
    average_speed = (displacement / total_time) if total_time != 0 else np.nan

    return {
        "particle_id": particle_id,
        "total_displacement_(um)": displacement,
        "total_time_(s)": total_time,
        "average_speed_(um/s)": average_speed,
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
        replicate, sample = td
        csv_path = os.path.join(TRACKING_DATA_DIR, td[0], td[1], "tracking_derived.csv")
        df = pl.read_csv(csv_path)
        particle_ids = (
            df.select(pl.col("particle"))
            .unique()
            .sort("particle")
            .to_series()
            .to_list()
        )

        particle_average_speeds = []
        for particle_id in particle_ids:
            particle_average_speeds.append(_get_particle_data(particle_id, df))

        particle_average_speeds_df = pl.DataFrame(particle_average_speeds)
        particle_average_speeds_df.write_csv(
            os.path.join(TRACKING_DATA_DIR, td[0], td[1], "average_speed.csv")
        )

    Parallel(n_jobs=-1)(
        delayed(process_tracking_data)(td)
        for td in tqdm(tracking_data, desc="Calculating average speed")
    )


if __name__ == "__main__":
    process_all_data()
