import re
import matplotlib.pyplot as plt
import os
import trackpy as tp
import zarr
import json
import pandas as pd
import seaborn as sns
import argparse
import constants

TRACKING_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "tracking_data"
)

PIXEL_SIZE = constants.PIXEL_SIZE


def extract_test_number(s):
    match = re.search(r"_test(\d+)", s)
    return int(match.group(1)) if match else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=str,
        help="Experiment name",
        default="FastTimeLaps_exp_02_AstChy1_110120_rep1",
    )
    args = parser.parse_args()

    exp = args.experiment

    # experiments = sorted(
    #     dir
    #     for dir in os.listdir(TRACKING_DATA_DIR)
    #     if os.path.isdir(os.path.join(TRACKING_DATA_DIR, dir))
    # )

    samples = [
        dir
        for dir in os.listdir(os.path.join(TRACKING_DATA_DIR, exp))
        if os.path.isdir(os.path.join(TRACKING_DATA_DIR, exp, dir))
    ]

    samples = sorted(samples, key=extract_test_number)

    plot_data = []

    for sample in samples:
        track_df = pd.read_csv(
            os.path.join(TRACKING_DATA_DIR, exp, sample, "tracking.csv")
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

        plot_data.append(data_dict)

    plot_df = pd.concat([pd.DataFrame(data) for data in plot_data])
    plot_df["test"] = plot_df["test"].astype(int)
    plot_df = plot_df.sort_values("test")
    plot_df["test"] = plot_df.apply(
        lambda row: f"{row['test']} | {row['step_init_abs']} - {row['step_end_abs']}",
        axis=1,
    )

    g = sns.FacetGrid(
        plot_df,
        col="test",
        height=1.25,
        col_wrap=10,
        sharex=True,
        sharey=True,
        hue="test",
    )

    def lineplot(x, y, **kwargs):
        sns.lineplot(x=x, y=y, **kwargs)

    g.map(lineplot, "lag_time", "msd")
    g.figure.tight_layout(w_pad=1)
    g.figure.suptitle(exp, fontsize=16)
    g.figure.text(
        0.5,
        0.95,
        "Ensemble Mean Squared Displacement",
        ha="center",
        va="center",
        fontsize=10,
    )

    g.figure.subplots_adjust(top=0.89)
    plt.rc("font", size=7)
    plt.show()
