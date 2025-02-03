"""
Package all output tracking data into a Zip file. Primarily used for sharing tracking data with collaborators and for exporting to a Plotly data server.
"""

import shutil
import os
import argparse
from alive_progress import alive_bar

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "output")
TRACKING_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "tracking-data"
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data-dir",
    type=str,
    default=DATA_DIR,
    help="Path to the directory containing the particle tracking data",
)

parser.add_argument(
    "--tracking-dir",
    type=str,
    default=TRACKING_DIR,
    help="Directory where tracking data will be saved to",
)

args = parser.parse_args()

sample_data = [
    (replicate, sample)
    for replicate in os.listdir(args.data_dir)
    if os.path.isdir(os.path.join(args.data_dir, replicate))
    for sample in os.listdir(os.path.join(args.data_dir, replicate))
    if os.path.isdir(os.path.join(args.data_dir, replicate, sample))
]

with alive_bar(len(sample_data)) as bar:
    for replicate, sample in sample_data:
        emsd_csv_path = os.path.join(
            args.data_dir, replicate, sample, "tracking-data", "emsd.csv"
        )
        imsd_csv_path = os.path.join(
            args.data_dir, replicate, sample, "tracking-data", "imsd.csv"
        )
        particles_csv_path = os.path.join(
            args.data_dir, replicate, sample, "tracking-data", "particles.csv"
        )
        tracks_csv_path = os.path.join(
            args.data_dir, replicate, sample, "tracking-data", "tracks.csv"
        )

        output_dir_path = os.path.join(
            os.path.join(args.tracking_dir, replicate, sample)
        )
        os.makedirs(output_dir_path, exist_ok=True)

        if os.path.isfile(emsd_csv_path):
            shutil.copy(emsd_csv_path, output_dir_path)
        if os.path.isfile(imsd_csv_path):
            shutil.copy(imsd_csv_path, output_dir_path)
        if os.path.isfile(particles_csv_path):
            shutil.copy(particles_csv_path, output_dir_path)
        if os.path.isfile(tracks_csv_path):
            shutil.copy(tracks_csv_path, output_dir_path)
        bar()
