#! /usr/bin/env python

import argparse

import dask.array as da
import numpy as np
import pandas as pd
import trackpy as tp
import zarr
import zarr.codecs
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect objects")

    parser.add_argument("--raw-data-zarr", type=str, help="Path to raw data zarr")
    parser.add_argument(
        "--large-objects-zarr", type=str, help="Path to large objects zarr"
    )

    args = parser.parse_args()

    raw_data_zarr_path = args.raw_data_zarr
    large_objects_zarr_path = args.large_objects_zarr

    raw_da = da.from_zarr(raw_data_zarr_path)
    assert raw_da.ndim == 3, "Expected 2D time-series data"
    assert raw_da.shape[1] == 712
    assert raw_da.shape[2] == 712
    assert raw_da.dtype == "uint8"

    exclude_large_objects = da.from_zarr(large_objects_zarr_path)

    frames = raw_da[:, :, :].compute()
    exclude = exclude_large_objects

    # Fill exclusion areas using mean intensity
    # of the entire time series
    mean_intensity = frames.mean()
    frames[exclude] = mean_intensity

    f = tp.batch(frames, diameter=5, minmass=40, separation=3)
    f.to_csv("detection.csv", index=False)

    # save detection overlay
    detection_overlays = []

    for t in range(frames.shape[0]):
        overlay = __draw_detection_overlay(f[f.frame == t], frames[t])
        detection_overlays.append(da.from_array(overlay))

    detection_da = da.stack(detection_overlays)
    detection_da = detection_da.rechunk()

    array = zarr.create_array(
        store="detection.zarr",
        shape=detection_da.shape,
        dtype=detection_da.dtype,
        chunks=(1, 356, 356, 3),
        shards=(20, 712, 712, 3),
        compressors=zarr.codecs.BloscCodec(
            cname="zstd", clevel=5, shuffle=zarr.codecs.BloscShuffle.shuffle
        ),
        zarr_format=3,
        dimension_names=["t", "y", "x", "c"],
    )

    array[:] = detection_da
