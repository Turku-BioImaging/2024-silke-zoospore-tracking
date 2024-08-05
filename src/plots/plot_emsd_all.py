import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from joblib import Parallel, delayed

TRACKING_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "tracking_data"
)


def read_data(replicate, sample):
    track_df = pd.read_csv(
        os.path.join(TRACKING_DATA_DIR, replicate, sample, "tracking_derived.csv"),
        low_memory=False,
    )
    emsd_df = pd.read_csv(
        os.path.join(TRACKING_DATA_DIR, replicate, sample, "emsd.csv")
    )

    data_dict = {
        "replicate": replicate,
        "test": track_df["test"].iloc[0],
        "step_init_abs": track_df["step_init_abs"].iloc[0],
        "step_end_abs": track_df["step_end_abs"].iloc[0],
        "lag_time": emsd_df["lagt"],
        "msd": emsd_df["msd"],
    }

    return data_dict


if __name__ == "__main__":
    sample_data = [
        (experiment, sample)
        for experiment in os.listdir(TRACKING_DATA_DIR)
        if os.path.isdir(os.path.join(TRACKING_DATA_DIR, experiment))
        for sample in os.listdir(os.path.join(TRACKING_DATA_DIR, experiment))
        if os.path.isdir(os.path.join(TRACKING_DATA_DIR, experiment, sample))
    ]

    all_sample_data = Parallel(n_jobs=-1)(
        delayed(read_data)(exp, sample) for exp, sample in tqdm(sample_data)
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

    g = sns.FacetGrid(
        replicate_df,
        col="test",
        hue="replicate",
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
    plt.subplots_adjust(top=0.9)
    plt.show()
    g.figure.savefig("emsd_all.png", dpi=300)
