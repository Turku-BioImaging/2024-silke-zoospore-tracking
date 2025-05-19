// params.input_dir = '../data/nd2'
// params.output_dir = "${projectDir}/output"

// params.convert_script = "${projectDir}/bin/convert.py"
// params.exclusion_mask_script = "${projectDir}/bin/exclusion_mask.py"
// params.detect_objects_script = "${projectDir}/bin/detect_objects.py"
// params.link_objects_script = "${projectDir}/bin/link_objects.py"
// params.metrics_script = "${projectDir}/bin/metrics.py"
// params.tiff_data_script = "${projectDir}/bin/tiff_data.py"

workflow {

    rawDataChannel = Channel
        .fromPath("${params.rawDataDir}/*/*.nd2")
        .map { filePath ->
            def replicateName = file(filePath).parent.baseName
            def sampleName = file(filePath).baseName
            return [filePath, replicateName, sampleName]
        }
        .take(1)
        // .view()


    ConvertND2ToZarr(rawDataChannel)
    // make_exclusion_masks(convert_to_zarr.out[0], params.output_dir, params.exclusion_mask_script)
    // detect_objects(make_exclusion_masks.out[0], params.output_dir, params.detect_objects_script)
    // link_objects(detect_objects.out[0], params.output_dir, params.link_objects_script)
    // calculate_metrics(link_objects.out[0], params.output_dir, params.metrics_script)
    // save_tiff_data(calculate_metrics.out[0], params.output_dir, params.tiff_data_script)
}


process ConvertND2ToZarr {
    input:
    tuple path(nd2Path), val(replicateName), val(sampleName)

    output:
    tuple path(nd2Path), val(replicateName), val(sampleName), path("raw-data.zarr")


    script:
    """
    convert_nd2_to_zarr.py \
        --nd2-path ${nd2Path}
    """
}

process make_exclusion_masks {
    input:
    tuple val(replicate_name), val(sample_name)
    path output_dir
    path exclusion_mask_script

    output:
    tuple val(replicate_name), val(sample_name)
    path "${output_dir}/${replicate_name}/${sample_name}/image-data-zarr/large-objects.zarr/.zarray"
    path "${output_dir}/${replicate_name}/${sample_name}/image-data-zarr/large-objects.zarr/0.0.0"

    script:
    """
    python ${exclusion_mask_script} --output-dir ${output_dir} --replicate ${replicate_name} --sample ${sample_name}
    """
}

process detect_objects {
    input:
    tuple val(replicate_name), val(sample_name)
    path output_dir
    path detect_objects_script

    output:
    tuple val(replicate_name), val(sample_name)
    path "${output_dir}/${replicate_name}/${sample_name}/image-data-zarr/detection.zarr/.zarray"
    path "${output_dir}/${replicate_name}/${sample_name}/image-data-zarr/detection.zarr/0.0.0.0"
    path "${output_dir}/${replicate_name}/${sample_name}/tracking-data/detection.csv"

    script:
    """
    python ${detect_objects_script} --output-dir ${output_dir} --replicate ${replicate_name} --sample ${sample_name}
    """
}

process link_objects {
    input:
    tuple val(replicate_name), val(sample_name)
    path output_dir
    path link_objects_script

    output:
    tuple val(replicate_name), val(sample_name)
    path "${output_dir}/${replicate_name}/${sample_name}/image-data-zarr/linking.zarr/.zarray"
    path "${output_dir}/${replicate_name}/${sample_name}/image-data-zarr/linking.zarr/0.0.0.0"
    path "${output_dir}/${replicate_name}/${sample_name}/tracking-data/tracking.csv"

    script:
    """
    python ${link_objects_script} --output-dir ${output_dir} --replicate ${replicate_name} --sample ${sample_name}
    """
}

process calculate_metrics {
    input:
    tuple val(replicate_name), val(sample_name)
    path output_dir
    path metrics_script

    output:
    tuple val(replicate_name), val(sample_name)
    path "${output_dir}/${replicate_name}/${sample_name}/tracking-data/particles.csv"

    script:
    """
    PYTHONWARNINGS=ignore  python ${metrics_script} --data-dir ${output_dir} --replicate ${replicate_name} --sample ${sample_name}
    """
}

process save_tiff_data {
    input:
    tuple val(replicate_name), val(sample_name)
    path output_dir
    path tiff_data_script

    output:
    path "${output_dir}/${replicate_name}/${sample_name}/image-data-tiff/raw_data.tif"
    path "${output_dir}/${replicate_name}/${sample_name}/image-data-tiff/detection.tif"
    path "${output_dir}/${replicate_name}/${sample_name}/image-data-tiff/linking.tif"

    script:
    """
    python ${tiff_data_script} --output-dir ${output_dir} --replicate ${replicate_name} --sample ${sample_name}
    """
}
