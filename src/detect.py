import zarr
import trackpy as tp
from skimage import draw, color
import dask.array as da
import os
from zarr.hierarchy import Group

ZARR_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "silke-zoospore-data.zarr"
)

TRACKING_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "tracking_data"
)


def __draw_detection_overlay(df, frame):
    rgb = color.gray2rgb(frame)

    height, width = frame.shape

    for _, row in df.iterrows():
        rr, cc = draw.circle_perimeter(int(row.y), int(row.x), 7)
        valid = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
        rr, cc = rr[valid], cc[valid]
        rgb[rr, cc] = [0, 255, 0]

    return rgb


def detect_objects(
    replicate: str,
    experiment: str,
    zarr_path: str = ZARR_PATH,
    save_tracking_data: bool = True,
) -> None:
    root: Group = zarr.open_group(zarr_path, mode="a")

    raw_da = da.from_zarr(root[f"{replicate}/{experiment}/raw_data"])  # type: ignore
    assert raw_da.ndim == 4, "Expected 4D data"
    assert raw_da.shape[1] == 3, "Expected 3 channels."
    assert raw_da.shape[2] == 712
    assert raw_da.shape[3] == 712
    assert raw_da.dtype == "uint8"

    frames = raw_da[:, 2, :, :].compute()
    f = tp.batch(frames, 7, minmass=100, maxsize=12)

    if save_tracking_data is True:
        tracking_dir_path = os.path.join(TRACKING_DATA_DIR, replicate, experiment)
        if not os.path.isdir(tracking_dir_path):
            os.makedirs(tracking_dir_path)

        f.to_csv(os.path.join(tracking_dir_path, "detection.csv"))

    # save detection overlay
    detection_overlays = []

    for t in range(frames.shape[0]):
        overlay = __draw_detection_overlay(f[f.frame == t], frames[t])
        detection_overlays.append(da.from_array(overlay))  # type: ignore

    detection_da = da.stack(detection_overlays)  # type: ignore
    detection_da = detection_da.rechunk()
    detection_da.to_zarr(
        url=zarr_path, component=f"{replicate}/{experiment}/detection", overwrite=True
    )

    overlay_group = root[f"{replicate}/{experiment}/detection"]
    overlay_group.attrs.update(
        {
            "description": "Object detection. Green circles indicate possible zoospores.",
            "author": "Turku BioImaging",
            "detection_parameters": {"diameter": 7, "minmass": 100, "maxsize": 12},
        }
    )
