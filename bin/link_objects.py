#! /usr/bin/env python

import argparse

import dask.array as da
import numpy as np
import pandas as pd
import trackpy as tp
import zarr
from scipy.spatial import ConvexHull
from skimage import color, draw

np.random.seed(874)
tp.linking.Linker.MAX_SUB_NET_SIZE = 10000
tp.quiet()


def __draw_detection_overlay(
    df: pd.DataFrame, frame: np.ndarray, color_dict: dict
) -> np.ndarray:
    rgb = color.gray2rgb(frame)

    height, width = frame.shape

    for _, row in df.iterrows():
        rr, cc = draw.circle_perimeter(int(row.y), int(row.x), 5)
        valid = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
        rr, cc = rr[valid], cc[valid]
        particle = row.particle
        rgb[rr, cc] = color_dict[particle]

    return rgb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Link detected objects")

    parser.add_argument("--raw-data-zarr", type=str, help="Path to raw data zarr")
    parser.add_argument("--detection-csv", type=str, help="Path to detection csv")

    args = parser.parse_args()

    # root = zarr.open_group(zarr_path, mode="a")
    f = pd.read_csv(args.detection_csv)

    pred = tp.predict.NearestVelocityPredict(span=20)
    t = pred.link_df(f, search_range=35, memory=20, adaptive_stop=5, adaptive_step=0.95)
    t = tp.filter_stubs(t, threshold=30)
    t = t[t["mass"] <= 900]
    t = t[t["size"] <= 1.8]

    groups = t.groupby("particle")
    area_covered_df = pd.DataFrame()
    for name, group in groups:
        if len(group) >= 3:
            hull = ConvexHull(group[["x", "y"]])
            area = hull.volume
            area_covered_df = pd.concat(
                [area_covered_df, pd.DataFrame({"particle": [name], "area": [area]})]
            )

    # Filter out particles that don't cover enough area.
    # This will remove particles that have little to no movement.
    # This setting is important in reducing the low-level noise in the data.
    area_covered_df.set_index("particle", inplace=True)
    threshold = 15**2  # 15 pixels squared
    particles_to_keep = area_covered_df[area_covered_df["area"] > threshold].index

    t = t[t["particle"].isin(particles_to_keep)]
    t.to_csv("linking.csv", escapechar="\\")

    # create linking overlay
    color_dict = {
        particle: tuple(np.random.randint(0, 256, 3))
        for particle in t["particle"].unique()
    }

    raw_da = da.from_zarr(args.raw_data_zarr)
    assert raw_da.ndim == 3, "Expected 2D time-series data"
    assert raw_da.shape[1] == 712
    assert raw_da.shape[2] == 712
    assert raw_da.dtype == "uint8"

    frames = raw_da[:, :, :].compute()

    overlay_frames = []
    for time in range(frames.shape[0]):
        overlay = __draw_detection_overlay(
            t[t["frame"] == time], frames[time], color_dict
        )
        overlay_frames.append(da.from_array(overlay))

    overlay_da = da.stack(overlay_frames)
    overlay_da = overlay_da.rechunk()

    array = zarr.create_array(
        store="linking.zarr",
        shape=overlay_da.shape,
        dtype=overlay_da.dtype,
        chunks=(1, 356, 356, 3),
        shards=(20, 712, 712, 3),
        compressors=zarr.codecs.BloscCodec(
            cname="zstd", clevel=5, shuffle=zarr.codecs.BloscShuffle.shuffle
        ),
        zarr_format=3,
        dimension_names=["t", "y", "x", "c"],
    )

    array[:] = overlay_da
