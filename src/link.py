import numpy as np
import os
import zarr
import trackpy as tp
import pandas as pd
from scipy.spatial import ConvexHull
from skimage import draw, color
from zarr.hierarchy import Group
import dask.array as da

ZARR_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "silke-zoospore-data.zarr"
)
TRACKING_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "tracking_data"
)

np.random.seed(874)
tp.linking.Linker.MAX_SUB_NET_SIZE = 10000


def __draw_detection_overlay(df, frame, color_keys: dict):
    rgb = color.gray2rgb(frame)

    height, width = frame.shape

    for _, row in df.iterrows():
        rr, cc = draw.circle_perimeter(int(row.y), int(row.x), 7)
        valid = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
        rr, cc = rr[valid], cc[valid]
        particle = row.particle
        rgb[rr, cc] = color_keys[particle]

    return rgb


def __validate_linking_dataset(root: Group, replicate: str, experiment: str) -> bool:
    if replicate not in root:
        return False
    if experiment not in root[replicate]:
        return False
    if "linking" not in root[replicate][experiment]:
        return False

    dataset = root[f"{replicate}/{experiment}/linking"]
    if "author" not in dataset.attrs:
        return False
    if dataset.attrs.get("author") != "Turku BioImaging":
        return False

    return True


def __validate_csv(replicate: str, experiment: str) -> bool:
    try:
        tracking_path = os.path.join(
            TRACKING_DATA_DIR, replicate, experiment, "tracking.csv"
        )
        df = pd.read_csv(tracking_path)
        assert len(df) > 0
        return True
    except Exception as _:
        return False


def link_detections(
    replicate: str,
    experiment: str,
    zarr_path: str = ZARR_PATH,
    overwrite: bool = False,
) -> None:
    root: Group = zarr.open_group(zarr_path, mode="a")

    valid_linking = __validate_linking_dataset(root, replicate, experiment)
    valid_csv = __validate_csv(replicate, experiment)

    if valid_linking and valid_csv and not overwrite:
        return

    detection_path = os.path.join(
        TRACKING_DATA_DIR, replicate, experiment, "detection.csv"
    )
    f = pd.read_csv(detection_path)

    tp.quiet()
    pred = tp.predict.NearestVelocityPredict(span=20)
    # t = pred.link_df(f, search_range=10, memory=20)
    # t = tp.link_df(f, search_range=10, memory=20)
    t = pred.link_df(f, search_range=12, memory=20, adaptive_stop=5, adaptive_step=0.95)
    t = tp.filter_stubs(t, 30)  # the min number of frames a particle must be present
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
    threshold = 10
    particles_to_keep = area_covered_df[area_covered_df["area"] > threshold].index

    t = t[t["particle"].isin(particles_to_keep)]
    t.to_csv(
        os.path.join(TRACKING_DATA_DIR, replicate, experiment, "tracking.csv"),
        escapechar="\\",
    )

    # create tracking overlay image
    color_dict = {
        particle: tuple(np.random.randint(0, 256, 3))
        for particle in t["particle"].unique()
    }

    raw_da = da.from_zarr(root[f"{replicate}/{experiment}/raw_data"])  # type: ignore
    assert raw_da.ndim == 4, "Expected 4D data"
    assert raw_da.shape[1] == 3, "Expected 3 channels."
    assert raw_da.shape[2] == 712
    assert raw_da.shape[3] == 712
    assert raw_da.dtype == "uint8"

    frames = raw_da[:, :, :].compute()

    overlay_frames = []

    for time in range(frames.shape[0]):
        overlay = __draw_detection_overlay(
            t[t["frame"] == time], frames[time], color_dict
        )
        overlay_frames.append(da.from_array(overlay))  # type: ignore

    overlay_da = da.stack(overlay_frames)  # type: ignore
    overlay_da = overlay_da.rechunk()
    overlay_da.to_zarr(
        url=zarr_path, component=f"{replicate}/{experiment}/linking", overwrite=True
    )

    overlay_group = root[f"{replicate}/{experiment}/linking"]
    overlay_group.attrs.update(
        {
            "description": "Frames with linked particles. Each color is a unique particle.",
            "author": "Turku BioImaging",
            "linking_parameters": {
                "predictor": "nearest velocity",
                "initial_velocity": 0.5,
                "search_range": 8,
                "memory": 20,
            },
            "filtering_parameters": {
                "mass_threshold": "<=630",
                "size_threshold": "<=1.8",
                "displacement_area": 5,
            },
        }
    )
