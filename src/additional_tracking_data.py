import numpy as np
import os
import trackpy as tp
import pandas as pd
from glob import glob
import re
import json
from tqdm import tqdm

TRACKING_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "tracking_data"
)

CLASSIFICATION_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "sample_classification.csv"
)

PIXEL_SIZE = 1.473175577212496


def classify_sample(sample: str, experiment: str) -> dict:
    # Get code of init light level
    index = sample.find("_from")
    assert index != -1, f"Invalid sample name: {sample}"
    step_init = int(sample[index + 5])
    step_init_abs = light_intensity_codes[str(step_init)]

    # Get code of end light level
    step_end = int(sample[index - 1])
    step_end_abs = light_intensity_codes[str(step_end)]

    # get step type
    if step_init_abs == step_end_abs:
        step_type = "adaptation"
    if step_init_abs < step_end_abs:
        step_type = "step_up"
    if step_init_abs > step_end_abs:
        step_type = "step_down"

    # Get species
    if "AstChy1" in experiment:
        species = "AstChy1"
    elif "StauChy1" in experiment:
        species = "StauChy1"
    else:
        raise ValueError(f"Could not infer species name: {experiment}/{sample}")

    # get replicate
    index = experiment.find("rep")
    replicate = int(experiment[index + 3])

    # get test number
    match = re.search("test\d{1,2}", sample)
    if match:
        digits = match.group()[4:]
        test = int(digits)
    else:
        raise ValueError(f"Could not infer test number: {experiment}/{sample}")

    # get frame interval
    if step_end == 5:
        frame_interval = 0.11237
    else:
        frame_interval = 0.02729

    return {
        "experiment": experiment,
        "sample": sample,
        "species": species,
        "replicate": replicate,
        "test": test,
        "step_init": step_init,
        "step_end": step_end,
        "step_init_abs": step_init_abs,
        "step_end_abs": step_end_abs,
        "step_type": step_type,
        "frame_interval": frame_interval,
    }


def calculate_speeds(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(by=["particle", "frame"])
    df["dx_(um)"] = df.groupby("particle")["x"].diff() * PIXEL_SIZE
    df["dy_(um)"] = df.groupby("particle")["y"].diff() * PIXEL_SIZE

    df["displacement_(um)"] = np.sqrt(df["dx_(um)"] ** 2 + df["dy_(um)"] ** 2)
    df["speed_(um/s)"] = df["displacement_(um)"] / df["frame_interval"]

    return df


if __name__ == "__main__":
    classification_df = pd.read_csv(CLASSIFICATION_DATA_PATH)

    with open(
        os.path.join(os.path.dirname(__file__), "light_intensity_codes.json"), "r"
    ) as f:
        light_intensity_codes = json.load(f)

    experiments = [
        dir
        for dir in os.listdir(TRACKING_DATA_DIR)
        if os.path.isdir(os.path.join(TRACKING_DATA_DIR, dir))
    ]

    for exp in experiments:
        print(f"Processing {exp}...")
        video_dirs = [
            dir
            for dir in os.listdir(os.path.join(TRACKING_DATA_DIR, exp))
            if os.path.isdir(os.path.join(TRACKING_DATA_DIR, exp, dir))
        ]

        for v in tqdm(video_dirs):
            df = pd.read_csv(os.path.join(TRACKING_DATA_DIR, exp, v, "tracking.csv"))

            if "Unnamed: 0" in df.columns:
                df = df.drop(columns=["Unnamed: 0"])

            sample_descriptors = classify_sample(v, exp)
            for col_name, data in sample_descriptors.items():
                df[col_name] = data

            df = calculate_speeds(df)
            df.to_csv(
                os.path.join(TRACKING_DATA_DIR, exp, v, "tracking.csv"), index=False
            )

        
