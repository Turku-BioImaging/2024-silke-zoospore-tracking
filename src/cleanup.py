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
from skimage.measure import label, regionprops
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
    if "exclusion_masks" in root[f"{replicate}/{experiment}"]:
        if "large_objects" in root[f"{replicate}/{experiment}/exclusion_masks"]:
            if overwrite is False:
                return

    large_objects = []

    for t in range(raw_da.shape[0]):
        thresholded = raw_da[t] > 50
        large = remove_small_objects(thresholded.compute(), min_size=30)

        labels = label(large)
        props = regionprops(labels)
        for prop in props:
            if prop.area > 3600:
                large[labels == prop.label] = 0

        large = dilation(large, footprint=np.ones((3, 3)))

        large_objects.append(da.from_array(large))

    large_objects = da.stack(large_objects)

    large_objects = large_objects.rechunk()
    large_objects.to_zarr(
        url=zarr_path,
        component=f"{replicate}/{experiment}/exclusion_masks/large_objects",
        overwrite=overwrite,
    )


if __name__ == "__main__":
    root = zarr.open(ZARR_PATH, mode="r")
    sample_data = [
        (replicate, experiment)
        for replicate in root.keys()
        for experiment in root[replicate].keys()
    ]

    for replicate, experiment in tqdm(sample_data):
        make_exclusion_masks(replicate, experiment, ZARR_PATH, overwrite=True)
