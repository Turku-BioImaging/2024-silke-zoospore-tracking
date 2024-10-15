"""
The Directionality Ratio is a measure of how straight the overall path of movement is, defined as the ratio of the net displacement to the total path length.
"""

import os
import polars as pl
from joblib import Parallel, delayed
from tqdm import tqdm
import constants

TRACKING_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "tracking_data"
)
PIXEL_SIZE = constants.PIXEL_SIZE
FRAME_INTERVAL_REGULAR = constants.FRAME_INTERVAL_REGULAR
FRAME_INTERVAL_LOW_LIGHT = constants.FRAME_INTERVAL_LOW_LIGHT


def _get_particle_data(particle_id: int, df: pl.DataFrame) -> dict:
    # particle id
    # Get first and last frames where the particle is present.
    # Get the net displacement of the particle.
    # Get the total path length of the particle.

    particle_data = df.filter(pl.col("particle") == particle_id)
    first_frame = particle_data.select(pl.col("frame")).min()
    last_frame = particle_data.select(pl.col("frame")).max()

    start_coords = particle_data.filter(pl.col("frame") == first_frame).select(
        ["x", "y"]
    )
    end_coords = particle_data.filter(pl.col("frame") == last_frame).select(["x", "y"])

    start_x, start_y = start_coords["x"][0], start_coords["y"][0]
    end_x, end_y = end_coords["x"][0], end_coords["y"][0]

    net_displacement = (
        (start_x - end_x) ** 2 + (start_y - end_y) ** 2
    ) ** 0.5 * PIXEL_SIZE

    total_displacement = particle_data.select(pl.col("displacement_(um)")).sum().item()

    return {
        "particle_id": particle_id,
        "total_displacement_(um)": total_displacement,
        "net_displacement_(um)": net_displacement,
        "directionality_ratio": net_displacement / total_displacement,
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
        csv_path = os.path.join(TRACKING_DATA_DIR, td[0], td[1], "tracking_derived.csv")
        df = pl.read_csv(csv_path)
        particle_ids = (
            df.select(pl.col("particle"))
            .unique()
            .sort("particle")
            .to_series()
            .to_list()
        )

        particle_drs = []

        for particle_id in particle_ids:
            particle_drs.append(_get_particle_data(particle_id, df))

        particle_drs_df = pl.DataFrame(particle_drs)
        particle_drs_df.write_csv(
            os.path.join(TRACKING_DATA_DIR, td[0], td[1], "directionality_ratios.csv")
        )

    Parallel(n_jobs=-1)(
        delayed(process_tracking_data)(td)
        for td in tqdm(tracking_data, desc="Calculating directionality ratios")
    )


if __name__ == "__main__":
    process_all_data()
