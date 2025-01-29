import os
import bioio_nd2
from bioio import BioImage
import dask.array as da
import zarr
import argparse


def convert_to_zarr(nd2_path: str, output_dir: str):
    img = BioImage(nd2_path, reader=bioio_nd2.Reader)

    img_da = img.dask_data
    img_da = da.squeeze(img_da, axis=(1, 2))
    img_da = da.moveaxis(img_da, -1, 1)
    img_da = img_da[:, 2, :, :]
    img_da = img_da.rechunk()

    metadata = dict(img.metadata)
    instrument = dict(metadata["instruments"][0])

    replicate_name = os.path.basename(os.path.dirname(nd2_path))
    sample_name = os.path.splitext(os.path.basename(nd2_path))[0]
    zarr_path = os.path.join(output_dir, replicate_name, sample_name, "image-data.zarr")

    os.makedirs(os.path.join(output_dir, replicate_name, sample_name), exist_ok=True)

    root = zarr.open(zarr_path, mode="a")

    dataset = root.create_dataset(
        "raw_data",
        data=img_da.compute(),
        chunks=(20, 712, 712),
        dtype=img_da.dtype,
        overwrite=True,
    )

    attrs = {
        "author": "Silke Van den Wyngaert, University of Turku",
        "pixel_size_y": img.physical_pixel_sizes.Y,
        "pixel_size_x": img.physical_pixel_sizes.X,
        "pixel_size_unit": "micrometer",
        "model": dict(instrument["detectors"][0])["model"],
        "lens_na": dict(instrument["objectives"][0])["lens_na"],
        "nominal_magnification": dict(instrument["objectives"][0])[
            "nominal_magnification"
        ],
    }
    dataset.attrs.update(attrs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ND2 files to Zarr format")
    parser.add_argument("--nd2-path", type=str, help="Path to ND2 file")
    parser.add_argument("--output-dir", type=str, help="Output directory")

    args = parser.parse_args()
    convert_to_zarr(args.nd2_path, args.output_dir)
