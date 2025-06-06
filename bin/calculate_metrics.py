#! /usr/bin/env python

import argparse
import re

import numpy as np
import polars as pl
import trackpy as tp
from scipy.spatial import ConvexHull

PIXEL_SIZE = 1.473175577212496
FRAME_INTERVAL_REGULAR = 0.02729  # 36.6 fps
FRAME_INTERVAL_LOW_LIGHT = 0.11237  # 8.9 fps
DIRECTION_CHANGE_THRESHOLD = 25
LIGHT_INTENSITY_CODES = {"0": 95, "1": 90, "2": 85, "3": 78, "4": 64, "5": 4}


def __classify_sample(replicate: str, sample: str) -> dict:
    # Get code of init light level
    index = sample.find("_from")
    assert index != -1, f"Invalid sample name: {sample}"
    step_init = int(sample[index + 5])
    step_init_abs = LIGHT_INTENSITY_CODES[str(step_init)]

    # Get code of end light level
    step_end = int(sample[index - 1])
    step_end_abs = LIGHT_INTENSITY_CODES[str(step_end)]

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


def __calculate_speeds(df: pl.DataFrame) -> pl.DataFrame:
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
                "dx_(um)"
            ),
            ((pl.col("y") - pl.col("y").shift(1)).over("particle") * PIXEL_SIZE).alias(
                "dy_(um)"
            ),
        ]
    )

    df = df.with_columns(
        (pl.col("dx_(um)").pow(2) + pl.col("dy_(um)").pow(2))
        .sqrt()
        .alias("displacement_(um)"),
    )

    return df


def __calculate_average_speed(particle_df: pl.DataFrame) -> dict:
    frame_interval = particle_df.select(pl.col("frame_interval"))[0].item()
    first_frame = particle_df.select(pl.col("frame")).min()
    last_frame = particle_df.select(pl.col("frame")).max()
    total_time = (last_frame.item() - first_frame.item()) * frame_interval

    displacement = particle_df.select(pl.col("displacement_(um)")).sum().item()
    average_speed = (displacement / total_time) if total_time != 0 else np.nan

    return {"total_displacement": displacement, "average_speed": average_speed}


def __calculate_curvilinear_velocity(particle_df: pl.DataFrame) -> dict:
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


def __calculate_directionality_ratio(particle_df: pl.DataFrame) -> dict:
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


def __calculate_straight_line_velocity(particle_df: pl.DataFrame) -> float:
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


def __calculate_direction_change_frequency(particle_df: pl.DataFrame) -> float:
    coordinates = particle_df.select(["x", "y"]).to_numpy()
    directions = np.diff(coordinates, axis=0)
    angles = np.arctan2(directions[1:, 1], directions[1:, 0]) - np.arctan2(
        directions[:-1, 1], directions[:-1, 0]
    )

    angles = np.degrees(angles) % 360
    direction_changes = np.sum(np.abs(angles) > DIRECTION_CHANGE_THRESHOLD)
    frame_interval = particle_df.select(pl.col("frame_interval"))[0].item()
    total_time = frame_interval * (len(coordinates) - 1)
    direction_change_frequency = direction_changes / total_time

    return direction_change_frequency


def __calculate_equivalent_diameter(particle_df: pl.DataFrame) -> float:
    points = particle_df.select(["x", "y"])
    hull = ConvexHull(points.to_numpy())
    area = hull.volume * PIXEL_SIZE**2

    return 2 * np.sqrt(area / np.pi)


def __get_particle_data_row(particle_id: int, df: pl.DataFrame) -> dict:
    particle_df = df.filter(pl.col("particle") == particle_id)

    speed = __calculate_average_speed(particle_df)
    curvilinear_velocity = __calculate_curvilinear_velocity(particle_df)
    directionality_ratio = __calculate_directionality_ratio(particle_df)
    straight_line_velocity = __calculate_straight_line_velocity(particle_df)
    equivalent_diameter = __calculate_equivalent_diameter(particle_df)
    direction_change_frequency = __calculate_direction_change_frequency(particle_df)

    return {
        "particle_id": particle_id,
        "net_displacement_(um)": directionality_ratio["net_displacement"],
        "total_displacement_(um)": speed["total_displacement"],
        "average_speed_(um/s)": speed["average_speed"],
        "total_time_(s)": curvilinear_velocity["total_time"],
        "curvilinear_velocity_(um/s)": curvilinear_velocity["curvilinear_velocity"],
        "straight_line_velocity_(um/s)": straight_line_velocity,
        "directionality_ratio": directionality_ratio["directionality_ratio"],
        "equivalent_diameter_(um)": equivalent_diameter,
        "direction_change_frequency_(Hz)": direction_change_frequency,
    }


def _calculate_additional_tracking_data(replicate: str, sample: str):
    df = pl.read_csv("linking.csv")

    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0")

    sample_descriptors = __classify_sample(replicate, sample)
    for col_name, value in sample_descriptors.items():
        df = df.with_columns(pl.lit(value).alias(col_name))

    df = __calculate_speeds(df)

    # write emsd and emsd data
    if len(df) == 0:
        df = pl.DataFrame(
            {
                "replicate": [],
                "sample": [],
                "frame": [],
                "particle": [],
                "x": [],
                "y": [],
                "test": [],
                "step_init": [],
                "step_end": [],
                "step_init_abs": [],
                "step_end_abs": [],
                "step_type": [],
                "frame_interval": [],
                "dx_(um)": [],
                "dy_(um)": [],
                "displacement_(um)": [],
            }
        )
        df.write_csv("tracks.csv")
        return

    fps = 1 / (df["frame_interval"][0])
    df_pandas = df.to_pandas()
    im = tp.imsd(df_pandas, mpp=PIXEL_SIZE, fps=fps, max_lagtime=450)
    im.to_csv("imsd.csv")

    em = tp.emsd(df_pandas, mpp=PIXEL_SIZE, fps=fps, max_lagtime=450)
    em.to_csv("emsd.csv")

    # write additional tracking data
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
            "dx_(um)",
            "dy_(um)",
            "displacement_(um)",
        ]
    )

    df.write_csv("additional-tracking-data.csv")


def _calculate_final_particle_tracking_data():
    df = pl.read_csv("additional-tracking-data.csv")
    particle_ids = (
        df.select(pl.col("particle")).unique().sort("particle").to_series().to_list()
    )

    particle_data = []
    for particle_id in particle_ids:
        particle_data.append(__get_particle_data_row(particle_id, df))

    particles_df = pl.DataFrame(particle_data)
    particles_df.write_csv("particles.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--replicate-name", type=str, required=True, help="Replicate name"
    )
    parser.add_argument("--sample-name", type=str, required=True, help="Sample name")
    parser.add_argument(
        "--linking-csv", type=str, required=True, help="Path to linking csv file"
    )
    args = parser.parse_args()

    _calculate_additional_tracking_data(args.replicate_name, args.sample_name)
    _calculate_final_particle_tracking_data()
