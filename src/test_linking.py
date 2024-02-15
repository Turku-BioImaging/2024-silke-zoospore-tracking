import numpy as np
import os
import zarr
import trackpy as tp
from skimage import draw, color
import argparse
import pandas as pd
from tqdm import tqdm

ZARR_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "silke-zoospore-data.zarr"
)
TRACKING_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "tracking_data"
)


def draw_detection_overlay(df, frame, color_keys: dict):
    rgb = color.gray2rgb(frame)

    height, width = frame.shape

    for _, row in df.iterrows():
        rr, cc = draw.circle_perimeter(int(row.y), int(row.x), 7)
        valid = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
        rr, cc = rr[valid], cc[valid]
        particle = row.particle
        rgb[rr, cc] = color_keys[particle]

    return rgb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing test_linking datasets",
    )
    parser.add_argument(
        "--save-tracking-data",
        type=bool,
        default=True,
        help="Save tracking data to CSV files",
    )

    args = parser.parse_args()
    overwrite = args.overwrite

    np.random.seed(874)
    root = zarr.open_group(ZARR_PATH, mode="a")

    experiments = sorted(root.keys())

    if overwrite is True:
        for exp in experiments:
            videos = root[exp].keys()
            for video in videos:
                if f"{exp}/{video}/test_linking" in root:
                    del root[f"{exp}/{video}/test_linking"]

    for exp in experiments:
        print(f"Processing {exp}...")
        videos = sorted(root[exp].keys())
        for video in tqdm(videos):
            if overwrite is False and f"{exp}/{video}/test_linking" in root:
                continue

            f_path = os.path.join(TRACKING_DATA_DIR, exp, video, "detection.csv")
            f = pd.read_csv(f_path)

            tp.quiet()
            pred = tp.predict.NearestVelocityPredict()
            t = pred.link_df(f, search_range=8, memory=15)
            t = tp.filter_stubs(t, 25)
            t = t[t["mass"] <= 1200]
            t = t[t["size"] <= 1.9]

            # Filter out particles with little or no movement
            t["displacement"] = np.sqrt(
                t.groupby("particle")["x"].diff() ** 2
                + t.groupby("particle")["y"].diff() ** 2
            )
            total_displacement = t.groupby("particle")["displacement"].sum()
            threshold = 25
            particles_to_keep = total_displacement[total_displacement > threshold].index
            t = t[t["particle"].isin(particles_to_keep)]

            color_dict = {
                particle: tuple(np.random.randint(0, 256, 3))
                for particle in t["particle"].unique()
            }

            # print(t.head())
            dataset = root[f"{exp}/{video}/raw_data"]
            assert dataset.ndim == 4
            assert dataset.shape[1] == 3
            assert dataset.shape[2] == 712
            assert dataset.shape[3] == 712
            assert dataset.dtype == np.uint8

            frames = dataset[:, 2, :, :]

            overlay_frames = []

            for time in range(frames.shape[0]):
                overlay = draw_detection_overlay(
                    t[t["frame"] == time], frames[time], color_dict
                )
                overlay_frames.append(overlay)

            overlay_frames = np.stack(overlay_frames)

            overlay_dataset = root.create_dataset(
                f"{exp}/{video}/test_linking", data=overlay_frames
            )

            attrs = {
                "description": "Frames with linked particles. Each color is a unique particle.",
                "author": "Turku BioImaging",
                "linking_parameters": {
                    "predictor": "nearest velocity",
                    "initial_velocity": 0.5,
                    "search_range": 8,
                    "memory": 15,
                    "mass_threshold": "<=630",
                    "size_threshold": "<=1.8",
                },
            }

            overlay_dataset.attrs.update(attrs)
