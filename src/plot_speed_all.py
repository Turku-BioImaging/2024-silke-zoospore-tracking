import os
import constants
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed

TRACKING_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "tracking_data"
)


def read_data(exp, sample):
    track_df = pd.read_csv(
        os.path.join(TRACKING_DATA_DIR, exp, sample, "tracking.csv"), low_memory=False
    )

    mean_speed_per_frame_df = (
        track_df.groupby("frame")["speed_(um/s)"].mean().reset_index()
    )

    median_speed_per_frame_df = (
        track_df.groupby("frame")["speed_(um/s)"].median().reset_index()
    )

    mean_speed_per_frame_df["lag_time"] = (
        mean_speed_per_frame_df["frame"] * track_df["frame_interval"].iloc[0]
    )

    data_dict = {
        "experiment": exp,
        "test": track_df["test"].iloc[0],
        "step_init_abs": track_df["step_init_abs"].iloc[0],
        "step_end_abs": track_df["step_end_abs"].iloc[0],
        # "frame": mean_speed_per_frame_df["frame"],
        "lag_time": mean_speed_per_frame_df["lag_time"],
        "mean_speed": mean_speed_per_frame_df["speed_(um/s)"],
        "median_speed": median_speed_per_frame_df["speed_(um/s)"],
    }

    return data_dict


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

        sample_data = Parallel(n_jobs=-1)(
            delayed(read_data)(exp, sample) for sample in samples
        )

        [all_exp_data.append(data) for data in sample_data]

    exp_df = pd.concat([pd.DataFrame(exp_data) for exp_data in all_exp_data])
    exp_df["test"] = exp_df["test"].astype(int)
    exp_df = exp_df.sort_values(by=["experiment", "test"])
    exp_df["test"] = exp_df.apply(
        lambda row: f"{row['test']} | {row['step_init_abs']} - {row['step_end_abs']}",
        axis=1,
    )

    # plot mean particle speed
    g = sns.FacetGrid(
        exp_df,
        col="test",
        hue="experiment",
        height=1.6,
        col_wrap=10,
        sharex=True,
        sharey=True,
    )

    g.map(sns.lineplot, "lag_time", "mean_speed").set(ylim=(0, 200))
    g.set_ylabels(r"Mean Speed ($\mu m/s$)")
    g.add_legend(borderaxespad=0.0, fontsize="x-small", ncol=1)
    g.figure.suptitle("Mean particle speed")
    plt.subplots_adjust(top=0.9)
    plt.show()
    g.figure.savefig("mean_speed_all.png", dpi=300)

    # plot median particle speed
    g = sns.FacetGrid(
        exp_df,
        col="test",
        hue="experiment",
        height=1.6,
        col_wrap=10,
        sharex=True,
        sharey=True,
    )

    g.map(sns.lineplot, "lag_time", "median_speed")
    g.set_ylabels(r"Median Speed ($\mu m/s$)")
    g.add_legend(borderaxespad=0.0, fontsize="x-small", ncol=1)
    g.figure.suptitle("Median particle speed")
    plt.subplots_adjust(top=0.9)
    plt.show()
    g.figure.savefig("median_speed_all.png", dpi=300)
