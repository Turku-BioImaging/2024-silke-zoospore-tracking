import argparse
import dask.array as da
import numpy as np
import zarr
from skimage.morphology import dilation, remove_small_objects
from skimage.measure import label, regionprops


def make_exclusion_mask(
    zarr_path: str,
    threshold_value: int = 50,
    object_min_size: int = 30,
    object_max_area: int = 3600,
    overwrite: bool = True,
):
    root = zarr.open(zarr_path, mode="a")
    raw_da = da.from_zarr(root["raw_data"])

    if "exclusion_masks" in root:
        if "large_objects" in root["exclusion_masks"]:
            if overwrite is False:
                return

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
    large_objects.to_zarr(
        url=zarr_path,
        component="exclusion_masks/large_objects",
        overwrite=overwrite,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create exclusion mask")
    parser.add_argument("--zarr-path", type=str, help="Path to Zarr file")
    parser.add_argument(
        "--threshold-value", type=int, default=50, help="Threshold value"
    )
    parser.add_argument(
        "--object-min-size", type=int, default=30, help="Minimum object size"
    )
    parser.add_argument(
        "--object-max-area", type=int, default=3600, help="Maximum object area"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing mask"
    )
    args = parser.parse_args()
    make_exclusion_mask(
        args.zarr_path,
        args.threshold_value,
        args.object_min_size,
        args.object_max_area,
        args.overwrite,
    )
