import numpy as np
import os
import zarr
import trackpy as tp
from skimage import draw, color
import argparse

ZARR_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "silke-zoospore-data.zarr"
)

TRACKING_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "tracking_data"
)


def draw_detection_overlay(df, frame):
    rgb = color.gray2rgb(frame)

    height, width = frame.shape

    for _, row in df.iterrows():
        rr, cc = draw.circle_perimeter(int(row.y), int(row.x), 7)
        valid = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
        rr, cc = rr[valid], cc[valid]
        rgb[rr, cc] = [0, 255, 0]

    return rgb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing test_detection datasets",
    )
    parser.add_argument(
        "--save-tracking-data",
        type=bool,
        default=True,
        help="Save tracking data to CSV files",
    )
    args = parser.parse_args()
    overwrite = args.overwrite

    root = zarr.open_group(ZARR_PATH, mode="a")

    experiments = root.keys()

    if overwrite is True:
        for exp in experiments:
            videos = root[exp].keys()
            for video in videos:
                if f"{exp}/{video}/test_detection" in root:
                    del root[f"{exp}/{video}/test_detection"]

    for exp in experiments:
        videos = root[exp].keys()
        for video in videos:
            if overwrite is False and f"{exp}/{video}/test_detection" in root:
                continue

            print(f"Processing {exp}/{video}")

            if f"{exp}/{video}/test_detection" in root:
                del root[f"{exp}/{video}/test_detection"]

            dataset = root[f"{exp}/{video}/raw_data"]
            assert dataset.ndim == 4, "Expected 4D dataset"
            assert dataset.shape[1] == 3, "Expected 3 channels"
            assert dataset.shape[2] == 712
            assert dataset.shape[3] == 712
            assert dataset.dtype == np.uint8

            frames = dataset[:, 2, :, :]
            f = tp.batch(frames, 7, minmass=100, maxsize=12)

            # Save detection dataframe to CSV
            if args.save_tracking_data is True:
                tracking_data_dir_path = os.path.join(TRACKING_DATA_DIR, exp, video)
                if not os.path.isdir(tracking_data_dir_path):
                    os.makedirs(tracking_data_dir_path)
                f.to_csv(os.path.join(tracking_data_dir_path, "detection.csv"))

            # Save detection overlay to Zarr dataset
            overlays = []

            for t in range(frames.shape[0]):
                overlay = draw_detection_overlay(f[f.frame == t], frames[t])
                overlays.append(overlay)
            overlays = np.stack(overlays)

            overlay_dataset = root.create_dataset(
                f"{exp}/{video}/test_detection", data=overlays
            )

            attrs = {
                "description": "Detection test. Green circles indicate detected zoospores.",
                "author": "Turku BioImaging",
                "detection_parameters": {"diameter": 7, "minmass": 100, "maxsize": 12},
            }

            overlay_dataset.attrs.put(attrs)
