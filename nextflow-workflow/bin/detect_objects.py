import argparse
import os

import dask.array as da
import numpy as np
import pandas as pd
import trackpy as tp
import zarr
from skimage import color, draw

tp.quiet()


def __draw_detection_overlay(df: pd.DataFrame, frame: np.ndarray) -> np.ndarray:
    rgb = color.gray2rgb(frame)

    height, width = frame.shape

    for _, row in df.iterrows():
        rr, cc = draw.circle_perimeter(int(row.y), int(row.x), 5)
        valid = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
        rr, cc = rr[valid], cc[valid]
        rgb[rr, cc] = [0, 255, 0]

    return rgb


def detect_objects(
    zarr_path: str,
    overwrite: bool = True,
) -> None:
    root = zarr.open(zarr_path, mode="a")
    raw_da = da.from_zarr(root["raw_data"])
    assert raw_da.ndim == 3, "Expected 2D time-series data"
    assert raw_da.shape[1] == 712
    assert raw_da.shape[2] == 712
    assert raw_da.dtype == "uint8"

    exclude_large_objects = da.from_zarr(root["exclusion_masks/large_objects"])

    frames = raw_da[:, :, :]
    exclude = exclude_large_objects

    # Fill exclusion areas using mean intensity
    # of the entire time series
    mean_intensity = frames.mean().compute()
    frames[exclude] = mean_intensity

    f = tp.batch(frames.compute(), diameter=5, minmass=40, separation=3)

    tracking_data_path = os.path.join(os.path.dirname(zarr_path), "tracking_data")
    os.makedirs(tracking_data_path, exist_ok=True)
    f.to_csv(os.path.join(tracking_data_path, "detection.csv"), index=False)

    # save detection overlay
    detection_overlays = []
    frames = frames.compute()
    for t in range(frames.shape[0]):
        overlay = __draw_detection_overlay(f[f.frame == t], frames[t])
        detection_overlays.append(da.from_array(overlay))

    detection_da = da.stack(detection_overlays)
    detection_da = detection_da.rechunk()

    detection_da.to_zarr(url=zarr_path, component="detection", overwrite=overwrite)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect objects")
    parser.add_argument("--zarr-path", type=str, help="Path to Zarr file")
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing detection data"
    )
    args = parser.parse_args()

    detect_objects(args.zarr_path, args.overwrite)
