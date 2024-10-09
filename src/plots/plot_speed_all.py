import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from tqdm import tqdm

TRACKING_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "tracking_data"
)


def read_data(replicate, sample):
    track_df = pd.read_csv(
        os.path.join(TRACKING_DATA_DIR, replicate, sample, "tracking_derived.csv"),
        low_memory=False,
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
        "replicate": replicate,
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
    sample_data = [
        (replicate, sample)
        for replicate in os.listdir(TRACKING_DATA_DIR)
        if os.path.isdir(os.path.join(TRACKING_DATA_DIR, replicate))
        for sample in os.listdir(os.path.join(TRACKING_DATA_DIR, replicate))
        if os.path.isdir(os.path.join(TRACKING_DATA_DIR, replicate, sample))
    ]

    all_sample_data = Parallel(n_jobs=-1)(
        delayed(read_data)(replicate, sample) for replicate, sample in tqdm(sample_data)
    )

    replicate_df = pd.concat(
        [pd.DataFrame(sample_data) for sample_data in all_sample_data]
    )

    replicate_df["test"] = replicate_df["test"].astype(int)
    replicate_df = replicate_df.sort_values(by=["replicate", "test"])
    replicate_df["test"] = replicate_df.apply(
        lambda row: f"{row['test']} | {row['step_init_abs']} - {row['step_end_abs']}",
        axis=1,
    )

    # plot mean particle speed
    g = sns.FacetGrid(
        replicate_df,
        col="test",
        hue="replicate",
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
        replicate_df,
        col="test",
        hue="replicate",
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
