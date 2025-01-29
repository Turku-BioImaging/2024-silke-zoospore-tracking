params.input_dir = '../data/nd2-test'
params.output_dir = "${projectDir}/output"

params.convert_script = "${projectDir}/bin/convert.py"
params.exclusion_mask_script = "${projectDir}/bin/exclusion_mask.py"
params.detect_objects_script = "${projectDir}/bin/detect_objects.py"
params.link_objects_script = "${projectDir}/bin/link_objects.py"
params.metrics_script = "${projectDir}/bin/metrics.py"
params.tiff_data_script = "${projectDir}/bin/tiff_data.py"

workflow {

    nd2_files = Channel
        .fromPath("${params.input_dir}/*/*.nd2")
        .map { file_path ->
            def replicate_name = file(file_path).parent.baseName
            return [file_path, replicate_name]
        }


    convert_to_zarr(nd2_files, params.output_dir, params.convert_script)
    make_exclusion_masks(convert_to_zarr.out[0], params.output_dir, params.exclusion_mask_script)
    detect_objects(make_exclusion_masks.out[0], params.output_dir, params.detect_objects_script)
    link_objects(detect_objects.out[0], params.output_dir, params.link_objects_script)
    calculate_metrics(link_objects.out[0], params.output_dir, params.metrics_script)
    save_tiff_data(calculate_metrics.out[0], params.output_dir, params.tiff_data_script)
}


process convert_to_zarr {
    input:
    tuple val(nd2_path), val(replicate_name)
    path output_dir
    path convert_script

    output:
    tuple val(replicate_name), val("${nd2_path.simpleName}"), val("${output_dir}/${replicate_name}/${nd2_path.simpleName}/image-data-zarr/raw-data.zarr")
    path "${output_dir}/${replicate_name}/${nd2_path.simpleName}/image-data-zarr/raw-data.zarr"
    path "${output_dir}/${replicate_name}/${nd2_path.simpleName}/image-data-zarr/raw-data.zarr/.zattrs"
    path "${output_dir}/${replicate_name}/${nd2_path.simpleName}/image-data-zarr/raw-data.zarr/0.0.0"

    script:
    """
    PYTHONWARNINGS=ignore python ${convert_script} --nd2-path ${nd2_path} --output-dir ${output_dir}
    """
}

process make_exclusion_masks {
    input:
    tuple val(replicate_name), val(sample_name), val(raw_data_zarr_path)
    path output_dir
    path exclusion_mask_script

    output:
    tuple val(replicate_name), val(sample_name)
    path "${output_dir}/${replicate_name}/${sample_name}/image-data-zarr/large-objects.zarr"
    path "${output_dir}/${replicate_name}/${sample_name}/image-data-zarr/large-objects.zarr/.zarray"
    path "${output_dir}/${replicate_name}/${sample_name}/image-data-zarr/large-objects.zarr/0.0.0"

    script:
    """
    python ${exclusion_mask_script} --raw-data-zarr-path ${raw_data_zarr_path} --output-dir ${output_dir} --replicate ${replicate_name} --sample ${sample_name}
    """
}

process detect_objects {
    input:
    tuple val(replicate_name), val(sample_name)
    path output_dir
    path detect_objects_script

    output:
    tuple val(replicate_name), val(sample_name)
    path "${output_dir}/${replicate_name}/${sample_name}/image-data-zarr/detection.zarr"
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
    path "${output_dir}/${replicate_name}/${sample_name}/image-data-zarr/linking.zarr/"
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
