import os

import constants
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from scipy.spatial import ConvexHull
from tqdm import tqdm

PIXEL_SIZE = constants.PIXEL_SIZE

TRACKING_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "tracking_data"
)


def read_data(replicate, sample):
    track_df = pd.read_csv(
        os.path.join(TRACKING_DATA_DIR, replicate, sample, "tracking_derived.csv"),
        low_memory=False,
    )

    particle_groups = track_df.groupby("particle")
    area_covered = pd.DataFrame()

    for name, group in particle_groups:
        if len(group) >= 3:
            hull = ConvexHull(group[["x", "y"]])
            area = hull.volume * PIXEL_SIZE**2
            area_covered = pd.concat(
                [
                    area_covered,
                    pd.DataFrame(
                        {
                            "replicate": replicate,
                            "sample": track_df["sample"].iloc[0],
                            "test": track_df["test"].iloc[0],
                            "step_init_abs": track_df["step_init_abs"].iloc[0],
                            "step_end_abs": track_df["step_end_abs"].iloc[0],
                            "particle": [name],
                            "area_(um^2)": [area],
                        }
                    ),
                ]
            )

    return area_covered


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

    g = sns.FacetGrid(
        replicate_df,
        col="test",
        hue="replicate",
        height=1.6,
        col_wrap=10,
        sharex=True,
        sharey=True,
    )

    g.map(
        sns.stripplot,
        "replicate",
        "area_(um^2)",
        order=replicate_df["replicate"].unique(),
    )
    g.set_ylabels(r"Area ($\mu m^2$)")
    g.set_xticklabels([])

    g.add_legend(
        borderaxespad=0.0,
        fontsize="small",
        ncol=1,
    )
    g.figure.suptitle("Area coverage (um^2)")
    plt.subplots_adjust(top=0.9)
    plt.show()
    g.figure.savefig("area_coverage_all.png", dpi=300)
