"""
Converts ND2 raw data to Zarr format
"""

import os
from glob import glob

import bioio_nd2
import dask.array as da
import zarr
from bioio import BioImage
from joblib import Parallel, delayed
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "nd2")
ZARR_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "silke-zoospore-data.zarr"
)


def process_sample(sample_name, overwrite=False):
    if not overwrite:
        if f"{parent_dir}/{sample_name}/raw_data" in root:
            return

    img = BioImage(
        os.path.join(DATA_DIR, parent_dir, f"{sample_name}.nd2"),
        reader=bioio_nd2.Reader,
    )

    img_da = img.dask_data
    img_da = da.squeeze(img_da, axis=(1, 2))
    img_da = da.moveaxis(img_da, -1, 1)
    img_da = img_da[:, 2, :, :]
    img_da = img_da.rechunk()

    metadata = dict(img.metadata)
    instrument = dict(metadata["instruments"][0])

    print(f"Writing {parent_dir}/{sample_name}")

    dataset = root.create_dataset(
        f"{parent_dir}/{sample_name}/raw_data",
        data=img_da.compute(),
        chunks=(20, 712, 712),
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
    parent_dirs = [
        d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))
    ]

    root = zarr.open(ZARR_PATH, mode="a")

    for parent_dir in parent_dirs:
        sample_names = [
            os.path.basename(f).replace(".nd2", "")
            for f in glob(os.path.join(DATA_DIR, parent_dir, "*.nd2"))
        ]

        Parallel(n_jobs=-1)(
            delayed(process_sample)(sample_name) for sample_name in tqdm(sample_names)
        )
