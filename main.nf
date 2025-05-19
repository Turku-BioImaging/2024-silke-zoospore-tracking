
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
    MakeExclusionMasks(ConvertND2ToZarr.out)
    DetectObjects(MakeExclusionMasks.out)
    LinkObjects(DetectObjects.out[0], DetectObjects.out[1])
    // calculate_metrics(link_objects.out[0], params.output_dir, params.metrics_script)
    // save_tiff_data(calculate_metrics.out[0], params.output_dir, params.tiff_data_script)
}


process ConvertND2ToZarr {
    publishDir "${params.outputDir}/${replicateName}/${sampleName}", mode: "copy", pattern: "raw-data.zarr"

    input:
    tuple path(nd2Path), val(replicateName), val(sampleName)

    output:
    tuple val(replicateName), val(sampleName), path("raw-data.zarr")

    script:
    """
    convert_nd2_to_zarr.py \
        --nd2-path ${nd2Path}
    """
}

process MakeExclusionMasks {
    publishDir "${params.outputDir}/${replicateName}/${sampleName}", mode: "copy", pattern: "large-objects.zarr"

    input:
    tuple val(replicateName), val(sampleName), path('raw-data.zarr')

    output:
    tuple val(replicateName), val(sampleName), path('raw-data.zarr'), path('large-objects.zarr')

    script:
    """
    make_exclusion_masks.py --zarr-path raw-data.zarr
    """
}

process DetectObjects {
    publishDir "${params.outputDir}/${replicateName}/${sampleName}", mode: "copy", pattern: "detection.{zarr,csv}"

    input:
    tuple val(replicateName), val(sampleName), path('raw-data.zarr'), path('large-objects.zarr')

    output:
    tuple val(replicateName), val(sampleName), path('raw-data.zarr'), path('large-objects.zarr'), emit: mainMetadata
    tuple path('detection.zarr'), path('detection.csv'), emit: detectionMetadata

    script:
    """
    detect_objects.py \
        --raw-data-zarr raw-data.zarr \
        --large-objects-zarr large-objects.zarr
    """
}

process LinkObjects {
    publishDir "${params.outputDir}/${replicateName}/${sampleName}", mode: "copy", pattern: "linking.{zarr,csv}"

    input:
    tuple val(replicateName), val(sampleName), path('raw-data.zarr'), path('large-objects.zarr')
    tuple path('detection.zarr'), path('detection.csv')

    output:
    tuple val(replicateName), val(sampleName), path('raw-data.zarr'), emit: mainMetadata    
    tuple path('linking.zarr'), path('linking.csv'), emit: linkingMetadata

    script:
    """
    link_objects.py \
        --raw-data-zarr raw-data.zarr \
        --detection-csv detection.csv
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
