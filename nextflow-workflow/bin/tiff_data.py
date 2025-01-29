import os
import zarr
from skimage import io
import dask.array as da
import argparse


def save_tiff_data(output_dir: str, replicate: str, sample: str) -> None:
    zarr_path = os.path.join(output_dir, replicate, sample, "image-data.zarr")
    tiff_dir = os.path.join(output_dir, replicate, sample, "image-data-tiff")

    os.makedirs(tiff_dir, exist_ok=True)

    root = zarr.open(zarr_path, mode="r")

    raw_da = da.from_zarr(root["raw_data"])
    detection_da = da.from_zarr(root["detection"])
    linking_da = da.from_zarr(root["linking"])

    io.imsave(
        os.path.join(tiff_dir, "raw_data.tif"), raw_da.compute(), check_contrast=False
    )
    io.imsave(
        os.path.join(tiff_dir, "detection.tif"),
        detection_da.compute(),
        check_contrast=False,
    )
    io.imsave(
        os.path.join(tiff_dir, "linking.tif"),
        linking_da.compute(),
        check_contrast=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, help="Path to the output directory.")
    parser.add_argument("--replicate", type=str, help="Name of the replicate.")
    parser.add_argument("--sample", type=str, help="Name of the sample.")

    args = parser.parse_args()

    save_tiff_data(args.output_dir, args.replicate, args.sample)
