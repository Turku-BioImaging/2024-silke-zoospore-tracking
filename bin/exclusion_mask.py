import os
import argparse
import dask.array as da
import numpy as np
from skimage.morphology import dilation, remove_small_objects
from skimage.measure import label, regionprops


def make_exclusion_mask(
    output_dir: str,
    replicate: str,
    sample: str,
    threshold_value: int = 50,
    object_min_size: int = 30,
    object_max_area: int = 3600,
):
    raw_data_zarr_path = os.path.join(
        output_dir, replicate, sample, "image-data-zarr", "raw-data.zarr"
    )
    raw_da = da.from_zarr(raw_data_zarr_path)

    large_objects = []
    for t in range(raw_da.shape[0]):
        thresholded = raw_da[t] > threshold_value
        large = remove_small_objects(thresholded.compute(), min_size=object_min_size)

        labels = label(large)
        props = regionprops(labels)
        for prop in props:
            if prop.area > object_max_area:
                large[labels == prop.label] = 0

        large = dilation(large, footprint=np.ones((3, 3)))
        large_objects.append(da.from_array(large))

    large_objects = da.stack(large_objects)
    large_objects = large_objects.rechunk()

    large_objects_zarr_path = os.path.join(
        output_dir, replicate, sample, "image-data-zarr", "large-objects.zarr"
    )

    large_objects.to_zarr(large_objects_zarr_path, overwrite=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create exclusion mask")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--replicate", type=str, help="Replicate name")
    parser.add_argument("--sample", type=str, help="Sample name")
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
    make_exclusion_mask(
        args.output_dir,
        args.replicate,
        args.sample,
        args.threshold_value,
        args.object_min_size,
        args.object_max_area,
    )
