import os
import matplotlib.pyplot as plt
import pandas as pd
import zarr
import json
import argparse
import constants
from tqdm import tqdm
import seaborn as sns
from joblib import Parallel, delayed


TRACKING_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "tracking_data"
)

PIXEL_SIZE = constants.PIXEL_SIZE


def read_data(exp, sample):
    track_df = pd.read_csv(
        os.path.join(TRACKING_DATA_DIR, exp, sample, "tracking.csv"), low_memory=False
    )
    emsd_df = pd.read_csv(os.path.join(TRACKING_DATA_DIR, exp, sample, "emsd.csv"))

    data_dict = {
        "experiment": exp,
        "test": track_df["test"].iloc[0],
        "step_init_abs": track_df["step_init_abs"].iloc[0],
        "step_end_abs": track_df["step_end_abs"].iloc[0],
        "lag_time": emsd_df["lagt"],
        "msd": emsd_df["msd"],
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

    g = sns.FacetGrid(
        exp_df,
        col="test",
        hue="experiment",
        height=1.6,
        col_wrap=10,
        sharex=True,
        sharey=True,
    )
    g.map(sns.lineplot, "lag_time", "msd")
    g.set_ylabels(r"MSD ($\mu m^2$)")
    g.add_legend(
        # bbox_to_anchor=(0, -0.2),
        # loc="upper left",
        borderaxespad=0.0,
        fontsize="small",
        ncol=1,
    )
    g.figure.suptitle("Ensemble Mean Squared Displacement")
    plt.show()
    g.figure.savefig("emsd_all.png", dpi=300)
