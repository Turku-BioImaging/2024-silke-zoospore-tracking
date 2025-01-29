import argparse
import os

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


def link_objects(zarr_path: str, overwrite: bool = True) -> None:
    root = zarr.open_group(zarr_path, mode="a")

    tracking_data_dir = os.path.join(os.path.dirname(zarr_path), "tracking_data")
    detection_csv_path = os.path.join(tracking_data_dir, "detection.csv")
    f = pd.read_csv(detection_csv_path)

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
    threshold = 15**2  # 25 pixels squared
    particles_to_keep = area_covered_df[area_covered_df["area"] > threshold].index

    t = t[t["particle"].isin(particles_to_keep)]
    t.to_csv(os.path.join(tracking_data_dir, "tracking.csv"), escapechar="\\")

    # create linking overlay
    color_dict = {
        particle: tuple(np.random.randint(0, 256, 3))
        for particle in t["particle"].unique()
    }

    raw_da = da.from_zarr(root["raw_data"])
    assert raw_da.ndim == 3, "Expected 2D time-series data"
    assert raw_da.shape[1] == 712
    assert raw_da.shape[2] == 712
    assert raw_da.dtype == "uint8"

    frames = raw_da[:, :, :].compute()

    overlay_frames = []
    for time in range(frames.shape[0]):
        overlay = __draw_detection_overlay(t[t["frame"] == time], frames[time], color_dict)
        overlay_frames.append(da.from_array(overlay))

    overlay_da = da.stack(overlay_frames)
    overlay_da = overlay_da.rechunk()
    overlay_da.to_zarr(url=zarr_path, component="linking", overwrite=overwrite)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Link detected objects")
    parser.add_argument("--zarr-path", type=str, help="Path to Zarr file")
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing data"
    )

    args = parser.parse_args()
    link_objects(args.zarr_path, args.overwrite)
