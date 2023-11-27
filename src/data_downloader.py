"""
Downloads random files from the CSC Allas container and converts ND2 files to generic TIFF.
Author: Junel Solis, Turku BioImaging, Turku, Finland

"""
import os
import swiftclient
from dotenv import load_dotenv
import numpy as np
import os
import random
from tqdm import tqdm
from aicsimageio import AICSImage
from glob import glob
from skimage import io

random.seed(858)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "nd2")
TIFF_DIR = os.path.join(os.path.dirname(__file__), "data", "tiff")


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

# Download random files from the container
os.makedirs(DATA_DIR, exist_ok=True)

file_paths = random.choices(conn.get_container("zoospore-time-lapse-data")[1], k=10)
for fpath in tqdm(file_paths):
    if not os.path.exists(os.path.join(DATA_DIR, fpath["name"])):
        obj = conn.get_object("zoospore-time-lapse-data", fpath["name"])[1]

        dir_path = fpath["name"].split("/")[0]
        os.makedirs(os.path.join(DATA_DIR, dir_path), exist_ok=True)

        with open(os.path.join(DATA_DIR, fpath["name"]), "wb") as f:
            f.write(obj)


# Convert ND2 files to TIFF
nd2_paths = glob(os.path.join(DATA_DIR, "**", "*.nd2"), recursive=True)
for p in tqdm(nd2_paths):
    img = AICSImage(p)
    img = np.squeeze(img.data, axis=(1, 2))
    img = np.moveaxis(img, -1, 1)

    dir_path = os.path.dirname(p).replace(DATA_DIR, TIFF_DIR).split("/")[-1]
    fname = os.path.basename(p).split(".")[0] + ".tif"

    if not os.path.exists(os.path.join(TIFF_DIR, dir_path)):
        os.makedirs(os.path.join(TIFF_DIR, dir_path))

    io.imsave(os.path.join(TIFF_DIR, dir_path, fname), img, check_contrast=False, imagej=True)
