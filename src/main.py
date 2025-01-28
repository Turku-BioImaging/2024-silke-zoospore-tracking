"""
Main entrypoint script for the entire processing pipeline. Expected input is a Zarr group containing
the raw data. The script will run object detection, linking, and metrics calculation.
"""

import os
import zarr
from cleanup import make_exclusion_masks
from detect import detect_objects
from link import link_detections
from metrics.particles import process_all_data as particle_metrics
from additional_tracking_data import process_all_tracks as add_tracking_data
import argparse
from alive_progress import alive_bar

ZARR_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "silke-zoospore-data.zarr"
)


def run_object_detection(replicate, experiment):
    detect_objects(replicate, experiment, ZARR_PATH)


def main(args):
    zarr_path = args.zarr_path
    root = zarr.open_group(zarr_path, mode="r")

    exp_data = [
        (replicate, experiment)
        for replicate in root.keys()
        for experiment in root[replicate].keys()
    ]

    if args.cleanup:
        with alive_bar(len(exp_data)) as bar:
            for replicate, experiment in exp_data:
                print(f"Creating masks for {replicate} -- {experiment}")
                make_exclusion_masks(replicate, experiment, zarr_path, args.overwrite)
                bar()

    if args.object_detection:
        with alive_bar(len(exp_data)) as bar:
            for replicate, experiment in exp_data:
                print(f"Detecting objects in {replicate} -- {experiment}")
                detect_objects(
                    replicate=replicate,
                    experiment=experiment,
                    zarr_path=zarr_path,
                    save_detection_data=True,
                    overwrite=args.overwrite,
                )
                bar()

    if args.linking:
        with alive_bar(len(exp_data)) as bar:
            for replicate, experiment in exp_data:
                print(f"Linking {replicate} -- {experiment}")
                link_detections(replicate, experiment, zarr_path, args.overwrite)
                bar()

    if args.metrics:
        add_tracking_data()
        particle_metrics()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr-path", type=str, default=ZARR_PATH)
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Generate exclusion masks to help clean up the raw data",
    )
    parser.add_argument("--object-detection", action="store_true")
    parser.add_argument("--linking", action="store_true")
    parser.add_argument("--metrics", action="store_true")
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing data"
    )
    args = parser.parse_args()

    main(args)
