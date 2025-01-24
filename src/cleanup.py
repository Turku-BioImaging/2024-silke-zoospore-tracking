"""
Module for cleaning raw data by removing non-moving and/or large particles. Depends
on global thresholding and binary morphological operations.
"""

import os

import dask.array as da
import numpy as np
import zarr

# from skimage.measure import label
from skimage.morphology import dilation, remove_small_objects
from tqdm import tqdm

ZARR_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "silke-zoospore-data.zarr"
)


def make_exclusion_masks(
    replicate: str, experiment: str, zarr_path: str = ZARR_PATH, overwrite: bool = True
):
    root = zarr.open(zarr_path, mode="a")

    raw_da = da.from_zarr(root[f"{replicate}/{experiment}/raw_data"])

    # large objects time-series
    large_objects = []

    for t in range(raw_da.shape[0]):
        thresholded = raw_da[t] > 50
        large = remove_small_objects(thresholded.compute(), min_size=30)
        large = dilation(large, footprint=np.ones((3, 3)))
        large_objects.append(da.from_array(large))

    large_objects = da.stack(large_objects)

    large_objects = large_objects.rechunk()
    large_objects.to_zarr(
        url=zarr_path,
        component=f"{replicate}/{experiment}/exclusion_masks/large_objects",
        overwrite=overwrite,
    )

    # # find non-moving small objects
    # # using MIP of time-series
    # non_moving = []
    # for t in range(raw_da.shape[0]):
    #     thresholded = raw_da[t] > 85
    #     non_moving.append(thresholded)

    # non_moving = da.stack(non_moving)
    # non_moving = da.max(non_moving, axis=0)
    # non_moving = non_moving.rechunk()

    # large_objects_mask = label(non_moving.compute())
    # large_objects_mask = remove_small_objects(large_objects_mask, min_size=50)
    # non_moving = non_moving * (large_objects_mask == 0)

    # non_moving.to_zarr(
    #     url=zarr_path,
    #     component=f"{replicate}/{experiment}/exclusion_masks/non_moving",
    #     overwrite=overwrite,
    # )


if __name__ == "__main__":
    root = zarr.open(ZARR_PATH, mode="r")
    sample_data = [
        (replicate, experiment)
        for replicate in root.keys()
        for experiment in root[replicate].keys()
    ]

    for replicate, experiment in tqdm(sample_data):
        make_exclusion_masks(replicate, experiment, ZARR_PATH, overwrite=True)
