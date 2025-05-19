#!/usr/bin/env python

import argparse

import dask.array as da
import numpy as np
import zarr
from skimage.measure import label, regionprops
from skimage.morphology import dilation, remove_small_objects

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create exclusion mask")
    parser.add_argument("--zarr-path", type=str, help="Path to raw data Zarr")
    parser.add_argument(
        "--threshold-value", type=int, default=50, help="Threshold value"
    )
    parser.add_argument(
        "--object-min-size", type=int, default=30, help="Minimum object size"
    )
    parser.add_argument(
        "--object-max-area", type=int, default=3600, help="Maximum object area"
    )

    args = parser.parse_args()

    raw_da = da.from_zarr(args.zarr_path)

    large_objects = []
    for t in range(raw_da.shape[0]):
        thresholded = raw_da[t] > args.threshold_value
        large = remove_small_objects(
            thresholded.compute(), min_size=args.object_min_size
        )

        labels = label(large)
        props = regionprops(labels)
        for prop in props:
            if prop.area > args.object_max_area:
                large[labels == prop.label] = 0

        large = dilation(large, footprint=np.ones((3, 3)))
        large_objects.append(da.from_array(large))

    large_objects = da.stack(large_objects)
    large_objects = large_objects.rechunk()

    array = zarr.create_array(
        store="large-objects.zarr",
        shape=large_objects.shape,
        dtype=large_objects.dtype,
        chunks=(1, 356, 356),
        shards=(20, 712, 712),
        compressors=zarr.codecs.BloscCodec(
            cname="zstd", clevel=5, shuffle=zarr.codecs.BloscShuffle.shuffle
        ),
        zarr_format=3,
        dimension_names=["t", "y", "x"],
    )

    array[:] = large_objects
