#! /usr/bin/env python

import argparse

import dask.array as da
from tifffile import imwrite

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw-data-zarr", required=True, type=str, help="Path to raw data zarr"
    )
    parser.add_argument(
        "--detection-zarr", required=True, type=str, help="Path to detection zarr"
    )
    parser.add_argument(
        "--linking-zarr", required=True, type=str, help="Path to linking zarr"
    )

    args = parser.parse_args()

    # save_tiff_data(args.output_dir, args.replicate, args.sample)
    raw_da = da.from_zarr(args.raw_data_zarr)
    detection_da = da.from_zarr(args.detection_zarr)
    linking_da = da.from_zarr(args.linking_zarr)

    imwrite("raw-data.tif", raw_da.compute(), compression="lzw")
    imwrite("detection.tif", detection_da.compute(), compression="lzw")
    imwrite("linking.tif", linking_da.compute(), compression="lzw")
