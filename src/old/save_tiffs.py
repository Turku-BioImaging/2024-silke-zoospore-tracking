"""
This script reads the Zarr group and saves TIFF versions.
"""
import os
import zarr
from skimage import io
import argparse
from tqdm import tqdm
from joblib import Parallel, delayed

ZARR_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "silke-zoospore-data.zarr"
)

TIFF_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "tiff_outputs")


def save_tiff(exp, sk, args):
    sample_dir = os.path.join(TIFF_DIR, exp, sk)
    os.makedirs(sample_dir, exist_ok=True)

    if args.raw_data:
        raw_data = root[exp][sk]["raw_data"]
        io.imsave(
            os.path.join(sample_dir, "raw_data.tif"),
            raw_data,
            check_contrast=False,
        )

    if args.detection:
        detection = root[exp][sk]["detection"]
        io.imsave(
            os.path.join(sample_dir, "detection.tif"),
            detection,
            check_contrast=False,
        )

    if args.linking:
        linking = root[exp][sk]["linking"]
        io.imsave(
            os.path.join(sample_dir, "linking.tif"),
            linking,
            check_contrast=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("")
    parser.add_argument("--raw-data", action="store_true", help="Save raw data.")
    parser.add_argument("--detection", action="store_true", help="Save detection data.")
    parser.add_argument("--linking", action="store_true", help="Save linking data.")

    args = parser.parse_args()

    root = zarr.open(ZARR_PATH, mode="r")

    experiments = sorted(list(root.keys()))
    for exp in experiments:
        sample_keys = sorted(list(root[exp].keys()))

        tasks = [(exp, sk, args) for sk in sample_keys]
        Parallel(n_jobs=4)(delayed(save_tiff)(*task) for task in tqdm(tasks))
