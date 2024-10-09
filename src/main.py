import os
import zarr
from detect import detect_objects
from link import link_detections
from tqdm import tqdm
from metrics.straight_line_velocity import process_all_data as straight_line_velocity
import argparse

ZARR_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "silke-zoospore-data.zarr"
)


def run_object_detection(replicate, experiment):
    detect_objects(replicate, experiment, ZARR_PATH)


def calculate_metrics():
    straight_line_velocity()


def main(args):
    root = zarr.open_group(ZARR_PATH, mode="r")

    exp_data = [
        (replicate, experiment)
        for replicate in root.keys()
        for experiment in root[replicate].keys()
    ]

    if args.object_detection:
        for replicate, experiment in tqdm(exp_data):
            detect_objects(replicate, experiment, ZARR_PATH)

    if args.linking:
        for replicate, experiment in tqdm(exp_data):
            link_detections(replicate, experiment, ZARR_PATH)

    if args.metrics:
        calculate_metrics()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-detection", action="store_true")
    parser.add_argument("--linking", action="store_true")
    parser.add_argument("--metrics", action="store_true")
    args = parser.parse_args()

    main(args)
