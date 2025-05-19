#! /usr/bin/env python


import argparse

import dask.array as da
import nd2
import zarr

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ND2 files to Zarr format")
    parser.add_argument("--nd2-path", type=str, help="Path to ND2 file")

    args = parser.parse_args()

    with nd2.ND2File(args.nd2_path) as f:
        img_da = f.to_dask()
        img_da = da.moveaxis(img_da, -1, 1)
        img_da = img_da[:, 2, :, :]

        zarr_path = "raw-data.zarr"

        array = zarr.create_array(
            store=zarr_path,
            shape=img_da.shape,
            dtype=img_da.dtype,
            chunks=(1, 356, 356),
            shards=(20, 712, 712),
            compressors=zarr.codecs.BloscCodec(
                cname="zstd", clevel=5, shuffle=zarr.codecs.BloscShuffle.shuffle
            ),
            zarr_format=3,
            dimension_names=["t", "y", "x"],
        )

        array[:] = img_da

        voxel_size = f.voxel_size()
        metadata = f.metadata.channels[0].microscope

        attrs = {
            "author": "Silke Van den Wyngaert, University of Turku",
            "pixel_size_y": voxel_size.y,
            "pixel_size_x": voxel_size.x,
            "pixel_size_unit": "micrometer",
            "microscope": {
                "objective_magnification": metadata.objectiveMagnification,
                "objective_name": metadata.objectiveName,
                "objective_numerical_aperture": metadata.objectiveNumericalAperture,
                "zoom_magnfication": metadata.zoomMagnification,
                "immersion_refractive_index": metadata.immersionRefractiveIndex,
                "modality": metadata.modalityFlags[0],
            },
        }

        array.attrs.update(attrs)
