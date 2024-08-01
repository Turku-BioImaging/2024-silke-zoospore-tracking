"""
Rechunk the zarr datasets using dask arrays. This will allow for larger chunks and fewer files.
"""

import os
import shutil
import zarr
import dask.array as da
from tqdm import tqdm
from zarr.hierarchy import Group
from joblib import Parallel, delayed

ZARR_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "silke-zoospore-data.zarr"
)

root: Group = zarr.open_group(ZARR_PATH, mode="r")

exp_data = [
    (replicate, exp)
    for replicate in list(root.keys())
    for exp in list(root[replicate].keys())
]


def _process_exp(replicate: str, exp: str) -> None:
    if "detection" in root[replicate][exp]:
        detection_da = da.from_zarr(root[replicate][exp]["detection"])  # type: ignore
        detection_da = detection_da.rechunk()
        detection_da.to_zarr(
            url=ZARR_PATH, component=f"{replicate}/{exp}/detection_da", overwrite=True
        )

    if "linking" in root[replicate][exp]:
        linking_da = da.from_zarr(root[replicate][exp]["linking"])  # type: ignore
        linking_da = linking_da.rechunk()
        linking_da.to_zarr(
            url=ZARR_PATH, component=f"{replicate}/{exp}/linking_da", overwrite=True
        )

    if "raw_data_da" in root[replicate][exp]:
        del root[f"{replicate}/{exp}/raw_data_da"]


def _cleanup_dirs(replicate: str, exp: str) -> None:
    # remove old detection and linking directories
    detection_path = os.path.join(ZARR_PATH, replicate, exp, "detection")
    new_detection_path = os.path.join(ZARR_PATH, replicate, exp, "detection_da")
    linking_path = os.path.join(ZARR_PATH, replicate, exp, "linking")
    new_linking_path = os.path.join(ZARR_PATH, replicate, exp, "linking_da")

    if os.path.isdir(detection_path) and os.path.isdir(new_detection_path):
        shutil.rmtree(detection_path)

    if os.path.isdir(linking_path) and os.path.isdir(new_linking_path):
        shutil.rmtree(linking_path)

    # rename the new detection and linking directories
    detection_path = os.path.join(ZARR_PATH, replicate, exp, "detection_da")
    linking_path = os.path.join(ZARR_PATH, replicate, exp, "linking_da")

    os.rename(detection_path, os.path.join(ZARR_PATH, replicate, exp, "detection"))
    os.rename(linking_path, os.path.join(ZARR_PATH, replicate, exp, "linking"))


# Parallel(n_jobs=1, verbose=10)(
#     delayed(_process_exp)(replicate, exp) for replicate, exp in exp_data
# )

Parallel(n_jobs=-1)(
    delayed(_cleanup_dirs)(replicate, exp) for replicate, exp in tqdm(exp_data)
)
