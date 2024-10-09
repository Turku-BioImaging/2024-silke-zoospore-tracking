import os
import pandas as pd
import zarr
from joblib import Parallel, delayed

TRACKING_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "tracking_data"
)


def count_rows(exp, sample):
    track_df = pd.read_csv(os.path.join(TRACKING_DATA_DIR, exp, sample, "tracking.csv"))
    return track_df.shape[0]


def count_particles(exp, sample):
    track_df = pd.read_csv(os.path.join(TRACKING_DATA_DIR, exp, sample, "tracking.csv"))

    return track_df["particle"].nunique()


if __name__ == "__main__":
    experiments = [
        exp
        for exp in os.listdir(TRACKING_DATA_DIR)
        if os.path.isdir(os.path.join(TRACKING_DATA_DIR, exp))
    ]

    total_rows = 0
    total_particles = 0
    for exp in experiments:
        print(f"Processing {exp}...")
        samples = [
            sample
            for sample in os.listdir(os.path.join(TRACKING_DATA_DIR, exp))
            if os.path.isdir(os.path.join(TRACKING_DATA_DIR, exp, sample))
        ]

        # count total number of rows
        row_counts = Parallel(n_jobs=-1)(
            delayed(count_rows)(exp, sample) for sample in samples
        )

        total_rows += sum(row_counts)

        # count total number of tracked particles
        particle_counts = Parallel(n_jobs=-1)(
            delayed(count_particles)(exp, sample) for sample in samples
        )

        total_particles += sum(particle_counts)

    print(f"Total rows: {total_rows}")
    print(f"Total particles: {total_particles}")
