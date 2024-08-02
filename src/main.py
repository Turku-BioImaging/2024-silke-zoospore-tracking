import os
import zarr
from detect import detect_objects
from link import link_detections
from tqdm import tqdm

ZARR_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "silke-zoospore-data.zarr"
)


def run_object_detection(replicate, experiment):
    detect_objects(replicate, experiment, ZARR_PATH)


def main():
    root = zarr.open_group(ZARR_PATH, mode="r")

    exp_data = [
        (replicate, experiment)
        for replicate in root.keys()
        for experiment in root[replicate].keys()
    ]

    # run object detection for possible zoospores
    for replicate, experiment in tqdm(exp_data):
        detect_objects(replicate, experiment, ZARR_PATH)

    # link detections (tracking)
    for replicate, experiment in tqdm(exp_data):
        link_detections(replicate, experiment, ZARR_PATH)


if __name__ == "__main__":
    main()
