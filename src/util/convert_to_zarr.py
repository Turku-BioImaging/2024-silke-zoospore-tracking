"""
Converts raw data to Zarr format
"""
import os

from multiprocessing import cpu_count
from glob import glob
import zarr
from aicsimageio import AICSImage as aic
import numpy as np
from joblib import Parallel, delayed

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "nd2")
ZARR_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "silke-zoospore-data.zarr"
)


def process_sample(sample_name):
    img = aic(os.path.join(DATA_DIR, parent_dir, f"{sample_name}.nd2"))
    img_data = img.data
    img_data = np.squeeze(img_data, axis=(1, 2))
    img_data = np.moveaxis(img_data, -1, 1)

    metadata = dict(img.metadata)
    instrument = dict(metadata["instruments"][0])

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

    print(f"Writing {parent_dir}/{sample_name}")
    dataset = root.create_dataset(
        f"{parent_dir}/{sample_name}/raw_data", data=img_data, chunks=(200, 1, 256, 256)
    )
    dataset.attrs.update(attrs)


if __name__ == "__main__":
    parent_dirs = [
        d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))
    ]

    root = zarr.open(ZARR_PATH, mode="w")

    n_jobs = max(cpu_count() - 4, 1)

    for parent_dir in parent_dirs:
        sample_names = [
            os.path.basename(f).replace(".nd2", "")
            for f in glob(os.path.join(DATA_DIR, parent_dir, "*.nd2"))
        ]

        Parallel(n_jobs=n_jobs)(
            delayed(process_sample)(sample_name) for sample_name in sample_names
        )
