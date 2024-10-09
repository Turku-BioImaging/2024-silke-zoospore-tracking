import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
import constants

TRACKING_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", '..', "data", "tracking_data"
)

DIRECTION_CHANGE_THRESHOLD = constants.DIRECTION_CHANGE_THRESHOLD


def calculate_freq_direction_change(exp, sample):
    track_df = pd.read_csv(
        os.path.join(TRACKING_DATA_DIR, exp, sample, "tracking.csv"), low_memory=False
    )

    particle_groups = track_df.groupby("particle")
    freq_direction_change = pd.DataFrame()

    for _, group in particle_groups:
        if len(group) >= 3:
            # calculate directions from coordinates
            coordinates = group[["x", "y"]].values
            directions = np.diff(coordinates, axis=0)
            angles = np.arctan2(directions[1:, 1], directions[1:, 0]) - np.arctan2(
                directions[:-1, 1], directions[:-1, 0]
            )

            angles = np.degrees(angles) % 360
            direction_changes = np.sum(np.abs(angles) > DIRECTION_CHANGE_THRESHOLD)
            frame_interval = group["frame_interval"].iloc[0]
            total_time = frame_interval * (len(coordinates) - 1)

            direction_change_freq = direction_changes / total_time
            freq_direction_change = pd.concat(
                [
                    freq_direction_change,
                    pd.DataFrame(
                        {
                            "experiment": exp,
                            "sample": [group["sample"].iloc[0]],
                            "test": [group["test"].iloc[0]],
                            "step_init_abs": [group["step_init_abs"].iloc[0]],
                            "step_end_abs": [group["step_end_abs"].iloc[0]],
                            "particle": [group["particle"].iloc[0]],
                            "direction_changes_per_time": [direction_change_freq],
                        }
                    ),
                ]
            )

    return freq_direction_change


if __name__ == "__main__":
    experiments = [
        exp
        for exp in os.listdir(TRACKING_DATA_DIR)
        if os.path.isdir(os.path.join(TRACKING_DATA_DIR, exp))
    ]

    all_exp_data = []
    for exp in experiments:
        print(f"Reading {exp}...")

        samples = [
            sample
            for sample in os.listdir(os.path.join(TRACKING_DATA_DIR, exp))
            if os.path.isdir(os.path.join(TRACKING_DATA_DIR, exp, sample))
        ]

        direction_change_freqs = Parallel(n_jobs=-1)(
            delayed(calculate_freq_direction_change)(exp, sample) for sample in samples
        )

        [all_exp_data.append(data) for data in direction_change_freqs]

    exp_df = pd.concat([pd.DataFrame(exp_data) for exp_data in all_exp_data])
    exp_df["test"] = exp_df["test"].astype(int)
    exp_df = exp_df.sort_values(by=["experiment", "test"])
    exp_df["test"] = exp_df.apply(
        lambda row: f"{row['test']} | {row['step_init_abs']} - {row['step_end_abs']}",
        axis=1,
    )

    g = sns.FacetGrid(
        exp_df,
        col="test",
        hue="experiment",
        height=1.6,
        col_wrap=10,
        sharex=True,
        sharey=True,
    )

    g.map(sns.stripplot, "experiment", "direction_changes_per_time")
    g.set_ylabels("Direction\nchanges per second")
    g.set_xticklabels([])
    g.add_legend(borderaxespad=0, fontsize="x-small", ncol=1)
    g.figure.suptitle("Frequency of direction changes per unit time")
    plt.subplots_adjust(top=0.9)
    plt.show()
    g.figure.savefig("direction_changes_per_time_all.png", dpi=300)
