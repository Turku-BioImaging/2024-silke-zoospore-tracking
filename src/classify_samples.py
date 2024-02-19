"""
Infer sample data from the experiment and video file names. Save information to a CSV file.
"""

import os
import zarr
import json
import pandas as pd
import re

ZARR_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "silke-zoospore-data.zarr"
)


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
    }


if __name__ == "__main__":
    root = zarr.open_group(ZARR_PATH, mode="r")

    experiments = sorted(root.keys())

    with open(
        os.path.join(os.path.dirname(__file__), "light_intensity_codes.json"), "r"
    ) as f:
        light_intensity_codes = json.load(f)

    data = []

    for exp in experiments:
        samples = sorted(root[exp].keys())

        for sample in samples:
            classification_dict = classify_sample(sample, exp)
            print(classification_dict)
            data.append(classification_dict)
            # break

        # break

    df = pd.DataFrame(data)
    df.to_csv("sample_classification.csv", index=False)
