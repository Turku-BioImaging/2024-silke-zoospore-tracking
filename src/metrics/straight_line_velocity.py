import os
import polars as pl
import constants
from tqdm import tqdm
from joblib import Parallel, delayed

TRACKING_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "tracking_data"
)

TRACKING_FNAME = "tracking_derived.csv"

PIXEL_SIZE = constants.PIXEL_SIZE
FRAME_INTERVAL_REGULAR = constants.FRAME_INTERVAL_REGULAR
FRAME_INTERVAL_LOW_LIGHT = constants.FRAME_INTERVAL_LOW_LIGHT


def _get_particle_data(particle_id: int, df: pl.DataFrame) -> dict:
    # particle id
    # Get first and last frames where the particle is present.
    # Get x,y coordinates of the particle in the first and last frames.
    # Get straight-line distance in microns between coords of first and last frames.
    # Calculate total time interval in seconds between first and last frames.
    particle_data = df.filter(pl.col("particle") == particle_id)
    frame_interval = df.select(pl.col("frame_interval"))[0].item()

    first_frame = particle_data.select(pl.col("frame")).min()
    last_frame = particle_data.select(pl.col("frame")).max()

    start_coords = particle_data.filter(pl.col("frame") == first_frame).select(
        ["x", "y"]
    )
    end_coords = particle_data.filter(pl.col("frame") == last_frame).select(["x", "y"])

    start_x, start_y = start_coords["x"][0], start_coords["y"][0]
    end_x, end_y = end_coords["x"][0], end_coords["y"][0]

    distance = ((start_x - end_x) ** 2 + (start_y - end_y) ** 2) ** 0.5 * PIXEL_SIZE

    total_time = (last_frame.item() - first_frame.item()) * frame_interval

    return {
        "particle_id": particle_id,
        "distance_(um)": distance,
        "total_time_(s)": total_time,
        "straight_line_velocity_(um/s)": distance / total_time,
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
        df = df.drop(
            [
                "replicate",
                "sample",
                "replicate_number",
                "species",
                "test",
                "step_init",
                "step_end",
                "step_init_abs",
                "step_type",
            ]
        )

        unique_particles = sorted(df.select("particle").unique().to_series().to_list())

        particle_sl_velocities = []

        for up in unique_particles:
            particle_sl_velocities.append(_get_particle_data(up, df))

        particle_sl_velocities_df = pl.DataFrame(particle_sl_velocities)

        particle_sl_velocities_df.write_csv(
            os.path.join(TRACKING_DATA_DIR, td[0], td[1], "straight_line_velocity.csv")
        )

    Parallel(n_jobs=-1)(
        delayed(process_tracking_data)(td)
        for td in tqdm(tracking_data, desc="Calculating straight line velocities")
    )


if __name__ == "__main__":
    process_all_data()
