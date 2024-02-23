"""
Renames the old test keys in the Zarr group.
"""
import os
import zarr
from tqdm import tqdm
import shutil

ZARR_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "silke-zoospore-data.zarr"
)

if __name__ == "__main__":
    root = zarr.open_group(ZARR_PATH, mode="a")

    experiments = sorted(list(root.keys()))

    key_mapping = {
        "test_detection": "detection",
        "test_linking": "linking",
    }

    for exp in experiments:
        sample_keys = sorted(list(root[exp].keys()))

        for sample in tqdm(sample_keys):
            for old_key, new_key in key_mapping.items():
                old_dir = os.path.join(ZARR_PATH, exp, sample, old_key)
                new_dir = os.path.join(ZARR_PATH, exp, sample, new_key)

                if os.path.exists(old_dir):
                    if os.path.exists(new_dir):
                        shutil.rmtree(new_dir)
                    os.rename(old_dir, new_dir)
                    # print(f"Renamed {old_dir} to {new_dir}")
