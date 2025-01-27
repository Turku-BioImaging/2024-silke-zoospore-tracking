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
    save_detection_data: bool = True,
    overwrite: bool = False,
) -> None:
    root: Group = zarr.open_group(zarr_path, mode="a")

    raw_da = da.from_zarr(root[f"{replicate}/{experiment}/raw_data"])
    assert raw_da.ndim == 3, "Expected 2D time-series data"
    assert raw_da.shape[1] == 712
    assert raw_da.shape[2] == 712
    assert raw_da.dtype == "uint8"

    exclude_large_objects = da.from_zarr(
        root[f"{replicate}/{experiment}/exclusion_masks/large_objects"]
    )

    # check if detection data already exists, skip if overwrite is False
    if overwrite is False and "detection" in root[f"{replicate}/{experiment}"]:
        return

    detection_overlays = []

    frames = raw_da[:, :, :].compute()
    exclude = exclude_large_objects.compute()

    frames[exclude] = 0

    tp.quiet()
    f = tp.batch(frames, 7, minmass=100, maxsize=12)

    if save_detection_data is True:
        tracking_dir_path = os.path.join(TRACKING_DATA_DIR, replicate, experiment)
        if not os.path.isdir(tracking_dir_path):
            os.makedirs(tracking_dir_path)

        f.to_csv(os.path.join(tracking_dir_path, "detection.csv"))

    # save detection overlay
    detection_overlays = []

    for t in range(frames.shape[0]):
        overlay = __draw_detection_overlay(f[f.frame == t], frames[t])
        detection_overlays.append(da.from_array(overlay))

    detection_da = da.stack(detection_overlays)
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
