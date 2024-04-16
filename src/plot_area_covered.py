import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from scipy.spatial import ConvexHull
import constants

PIXEL_SIZE = constants.PIXEL_SIZE

TRACKING_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "tracking_data"
)


def read_data(exp, sample):
    track_df = pd.read_csv(
        os.path.join(TRACKING_DATA_DIR, exp, sample, "tracking.csv"), low_memory=False
    )

    particle_groups = track_df.groupby("particle")
    area_covered = pd.DataFrame()

    for name, group in particle_groups:
        if len(group) >= 3:
            hull = ConvexHull(group[["x", "y"]])
            area = hull.volume * PIXEL_SIZE
            area_covered = pd.concat(
                [
                    area_covered,
                    pd.DataFrame(
                        {
                            "experiment": exp,
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

    # area_covered.set_index("particle", inplace=True)

    return area_covered


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

    # sample_df = exp_df.sample(n=50000, random_state=535)

    # print(exp_df.head())
    # print(exp_df.describe())

    g = sns.FacetGrid(
        exp_df,
        # sample_df,
        col="test",
        hue="experiment",
        height=1.6,
        col_wrap=10,
        sharex=True,
        sharey=True,
    )

    g.map(sns.stripplot, "experiment", "area_(um^2)")
    g.set_ylabels(r"Area ($\mu m^2$)")
    g.set_xticklabels([])
    # for ax in g.axes.flat:
    #     ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: "{:0.1e}".format(x)))
    g.add_legend(
        borderaxespad=0.0,
        fontsize="small",
        ncol=1,
    )
    g.figure.suptitle("Area coverage (um^2)")
    plt.subplots_adjust(top=0.9)
    plt.show()
    g.figure.savefig("area_coverage_all.png", dpi=300)
