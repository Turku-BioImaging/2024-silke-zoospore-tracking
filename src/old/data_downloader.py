"""
Downloads random files from the CSC Allas container and converts ND2 files to generic TIFF.
Author: Junel Solis, Turku BioImaging, Turku, Finland

"""
import os
import random
from glob import glob

import numpy as np
import swiftclient
from aicsimageio import AICSImage
from dotenv import load_dotenv
from skimage import io
from tqdm import tqdm
import multiprocessing as mp

random.seed(858)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "nd2")
# TIFF_DIR = os.path.join(os.path.dirname(__file__), "data", "tiff")
TIFF_DIR = os.path.join("/", "mnt", "f", "silke-convert-temp")

# EXCLUSION_LIST = [
#     "FastTimeLaps_exp_02_StauChy1_230120_rep3/FastTimeLaps_exp_test1_1_from0.nd2"
# ]


# Configure CSC Allas connection
load_dotenv()

_authurl = os.getenv("OS_AUTH_URL")
_auth_version = os.getenv("OS_IDENTITY_API_VERSION")
_user = os.getenv("OS_USERNAME")
_key = os.getenv("OS_PASSWORD")
_os_options = {"project_name": os.getenv("OS_PROJECT_NAME")}

conn = swiftclient.Connection(
    authurl=_authurl,
    user=_user,
    key=_key,
    os_options=_os_options,
    auth_version=_auth_version,
)

resp_headers, containers = conn.get_account()

os.makedirs(DATA_DIR, exist_ok=True)

# file_paths = random.choices(conn.get_container("zoospore-time-lapse-data")[1], k=10)
file_paths = conn.get_container("zoospore-time-lapse-data")[1]

for fpath in tqdm(file_paths):
    if not os.path.exists(os.path.join(DATA_DIR, fpath["name"])):
        obj = conn.get_object("zoospore-time-lapse-data", fpath["name"])[1]

        dir_path = fpath["name"].split("/")[0]
        os.makedirs(os.path.join(DATA_DIR, dir_path), exist_ok=True)

        # check that file is not write-protected
        if not os.access(os.path.join(DATA_DIR, fpath["name"]), os.W_OK):
            continue

        with open(os.path.join(DATA_DIR, fpath["name"]), "wb") as f:
            f.write(obj)


# Convert ND2 files to TIFF
def convert_nd2_to_tiff(p):
    dir_path = os.path.dirname(p).replace(DATA_DIR, TIFF_DIR).split("/")[-1]
    fname = os.path.basename(p).split(".")[0] + ".tif"

    if os.path.isfile(os.path.join(TIFF_DIR, dir_path, fname)):
        return

    if not os.path.exists(os.path.join(TIFF_DIR, dir_path)):
        os.makedirs(os.path.join(TIFF_DIR, dir_path), exist_ok=True)

    img = AICSImage(p)
    img = np.squeeze(img.data, axis=(1, 2))
    img = np.moveaxis(img, -1, 1)
    io.imsave(
        os.path.join(TIFF_DIR, dir_path, fname), img, check_contrast=False, imagej=True
    )


nd2_paths = glob(os.path.join(DATA_DIR, "**", "*.nd2"), recursive=True)
with mp.Pool() as pool:
    max_ = len(nd2_paths)
    with tqdm(total=max_) as pbar:
        for i, _ in enumerate(pool.imap_unordered(convert_nd2_to_tiff, nd2_paths)):
            pbar.update()
