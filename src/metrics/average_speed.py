import os
import polars as pl
from joblib import Parallel, delayed
from tqdm import tqdm
import constants
import numpy as np
from scipy.spatial import ConvexHull

TRACKING_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "tracking_data"
)
PIXEL_SIZE = constants.PIXEL_SIZE
FRAME_INTERVAL_REGULAR = constants.FRAME_INTERVAL_REGULAR
FRAME_INTERVAL_LOW_LIGHT = constants.FRAME_INTERVAL_LOW_LIGHT


def _calculate_average_speed(particle_df: pl.DataFrame) -> dict:
    frame_interval = particle_df.select(pl.col("frame_interval"))[0].item()
    first_frame = particle_df.select(pl.col("frame")).min()
    last_frame = particle_df.select(pl.col("frame")).max()
    total_time = (last_frame.item() - first_frame.item()) * frame_interval

    displacement = particle_df.select(pl.col("displacement_(um)")).sum().item()
    average_speed = (displacement / total_time) if total_time != 0 else np.nan

    return {"total_displacement": displacement, "average_speed": average_speed}


def _calculate_curvilinear_velocity(particle_df: pl.DataFrame) -> dict:
    displacement = particle_df.select(pl.col("displacement_(um)")).sum().item()
    frame_interval = particle_df.select(pl.col("frame_interval"))[0].item()
    first_frame = particle_df.select(pl.col("frame")).min()
    last_frame = particle_df.select(pl.col("frame")).max()
    total_time = (last_frame.item() - first_frame.item()) * frame_interval
    curvilinear_velocity = (displacement / total_time) if total_time != 0 else np.nan

    return {
        "total_time": total_time,
        "curvilinear_velocity": curvilinear_velocity,
    }


def _calculate_straight_line_velocity(particle_df: pl.DataFrame) -> float:
    frame_interval = particle_df.select(pl.col("frame_interval"))[0].item()
    first_frame = particle_df.select(pl.col("frame")).min()
    last_frame = particle_df.select(pl.col("frame")).max()
    start_coords = particle_df.filter(pl.col("frame") == first_frame).select(["x", "y"])
    end_coords = particle_df.filter(pl.col("frame") == last_frame).select(["x", "y"])

    start_x, start_y = start_coords["x"][0], start_coords["y"][0]
    end_x, end_y = end_coords["x"][0], end_coords["y"][0]

    distance = ((start_x - end_x) ** 2 + (start_y - end_y) ** 2) ** 0.5 * PIXEL_SIZE
    total_time = (last_frame.item() - first_frame.item()) * frame_interval

    return (distance / total_time) if total_time != 0 else np.nan


def _calculate_directionality_ratio(particle_df: pl.DataFrame) -> dict:
    first_frame = particle_df.select(pl.col("frame")).min()
    last_frame = particle_df.select(pl.col("frame")).max()

    start_coords = particle_df.filter(pl.col("frame") == first_frame).select(["x", "y"])
    end_coords = particle_df.filter(pl.col("frame") == last_frame).select(["x", "y"])

    start_x, start_y = start_coords["x"][0], start_coords["y"][0]
    end_x, end_y = end_coords["x"][0], end_coords["y"][0]

    net_displacement = (
        (start_x - end_x) ** 2 + (start_y - end_y) ** 2
    ) ** 0.5 * PIXEL_SIZE

    total_displacement = particle_df.select(pl.col("displacement_(um)")).sum().item()

    return {
        "net_displacement": net_displacement,
        "total_displacement": total_displacement,
        "directionality_ratio": (net_displacement / total_displacement)
        if total_displacement != 0
        else np.nan,
    }


def _calculate_area_covered(particle_df: pl.DataFrame) -> float:
    points = particle_df.select(["x", "y"])
    hull = ConvexHull(points.to_numpy())
    area = hull.volume * PIXEL_SIZE**2
    return area


def _get_particle_data(particle_id: int, df: pl.DataFrame) -> dict:
    particle_df = df.filter(pl.col("particle") == particle_id)

    speed = _calculate_average_speed(particle_df)
    curvilinear_velocity = _calculate_curvilinear_velocity(particle_df)
    directionality_ratio = _calculate_directionality_ratio(particle_df)
    straight_line_velocity = _calculate_straight_line_velocity(particle_df)
    area_covered = _calculate_area_covered(particle_df)

    return {
        "particle_id": particle_id,
        "net_displacement_(um)": directionality_ratio["net_displacement"],
        "total_displacement_(um)": speed["total_displacement"],
        "average_speed_(um/s)": speed["average_speed"],
        "total_time_(s)": curvilinear_velocity["total_time"],
        "curvilinear_velocity_(um/s)": curvilinear_velocity["curvilinear_velocity"],
        "straight_line_velocity_(um/s)": straight_line_velocity,
        "directionality_ratio": directionality_ratio["directionality_ratio"],
        "area_covered_(um^2)": area_covered,
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
        csv_path = os.path.join(TRACKING_DATA_DIR, td[0], td[1], "tracks.csv")
        df = pl.read_csv(csv_path)
        particle_ids = (
            df.select(pl.col("particle"))
            .unique()
            .sort("particle")
            .to_series()
            .to_list()
        )

        particle_data = []
        for particle_id in particle_ids:
            particle_data.append(_get_particle_data(particle_id, df))

        particles_df = pl.DataFrame(particle_data)
        particles_df.write_csv(
            os.path.join(TRACKING_DATA_DIR, td[0], td[1], "particles.csv")
        )

    Parallel(n_jobs=-1)(
        delayed(process_tracking_data)(td)
        for td in tqdm(tracking_data, desc="Calculating particle metrics")
    )

    # for replicate, sample in tracking_data[:1]:
    #     csv_path = os.path.join(TRACKING_DATA_DIR, replicate, sample, "tracks.csv")
    #     df = pl.read_csv(csv_path)
    #     particle_ids = (
    #         df.select(pl.col("particle"))
    #         .unique()
    #         .sort("particle")
    #         .to_series()
    #         .to_list()
    #     )

    #     for particle_id in particle_ids:
    #         _get_particle_data(particle_id, df)


if __name__ == "__main__":
    process_all_data()
