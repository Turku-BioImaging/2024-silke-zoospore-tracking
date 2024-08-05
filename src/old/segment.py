import os
from glob import glob
import numpy as np
from skimage import io, img_as_ubyte
from tqdm import tqdm

TIFF_DIR = os.path.join(os.path.dirname(__file__), "data", "tiff")
WORK_DIR = os.path.join(os.path.dirname(__file__), "workdir")

os.makedirs(WORK_DIR, exist_ok=True)

img_paths = glob(os.path.join(TIFF_DIR, "**", "*.tif"), recursive=True)

for p in tqdm(img_paths):
    img = io.imread(p)[:, :, :, 2]

    derivative_frames = []

    for t in range(1, img.shape[0]):
        der = img[t] - img[t - 1]
        der = np.where(np.abs(der) < 250, der, 0)
        derivative_frames.append(der)

    derivative_frames = np.array(derivative_frames)

    io.imsave(os.path.join(WORK_DIR, f"{os.path.basename(p)}"), derivative_frames)
    break
