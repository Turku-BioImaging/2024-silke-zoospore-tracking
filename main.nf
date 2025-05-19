workflow {

    rawDataChannel = Channel.fromPath("${params.rawDataDir}/*/*.nd2")
        .map { filePath ->
            def replicateName = file(filePath).parent.baseName
            def sampleName = file(filePath).baseName
            return [filePath, replicateName, sampleName]
        }
        .take(1)
    // .view()


    ConvertND2ToZarr(rawDataChannel).set { rawDataZarrChannel }
    MakeExclusionMasks(ConvertND2ToZarr.out)
    DetectObjects(MakeExclusionMasks.out).set { detectObjectsChannel }
    LinkObjects(DetectObjects.out).set { linkObjectsChannel }
    CalculateMetrics(LinkObjects.out)


    tiffDataChannel = rawDataZarrChannel
        .combine(detectObjectsChannel, by: [0, 1])
        .combine(linkObjectsChannel, by: [0, 1])
        .map { i ->
            def replicateName = i[0]
            def sampleName = i[1]
            def rawDataZarr = i[2]
            def detectionZarr = i[4]
            def linkingZarr = i[7]

            return tuple(replicateName, sampleName, rawDataZarr, detectionZarr, linkingZarr)
        }
        .view()

    SaveTiffData(tiffDataChannel)
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
    tuple val(replicateName), val(sampleName), path('raw-data.zarr'), path('detection.zarr'), path('detection.csv')

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
    tuple val(replicateName), val(sampleName), path('raw-data.zarr'), path('detection.zarr'), path('detection.csv')

    output:
    tuple val(replicateName), val(sampleName), path('raw-data.zarr'), path('linking.zarr'), path('linking.csv')

    script:
    """
    link_objects.py \
        --raw-data-zarr raw-data.zarr \
        --detection-csv detection.csv
    """
}

process CalculateMetrics {
    publishDir "${params.outputDir}/${replicateName}/${sampleName}", mode: 'copy', pattern: '*.csv'

    input:
    tuple val(replicateName), val(sampleName), path('raw-data.zarr'), path('linking.zarr'), path('linking.csv')

    output:
    tuple val(replicateName), val(sampleName), path('raw-data.zarr'), path('particles.csv'), path('emsd.csv'), path('imsd.csv')

    script:
    """
    calculate_metrics.py \
        --replicate-name ${replicateName} \
        --sample-name ${sampleName} \
        --linking-csv linking.csv
    """
}

process SaveTiffData {
    publishDir "${params.outputDir}/${replicateName}/${sampleName}", mode: 'copy', pattern: '*.tif'

    input:
    tuple val(replicateName), val(sampleName), path('raw-data.zarr'), path('detection.zarr'), path('linking.zarr')

    output:
    tuple path('raw-data.tif'), path('detection.tif'), path('linking.tif')

    script:
    """
    save_tiff_data.py \
        --raw-data-zarr raw-data.zarr \
        --detection-zarr detection.zarr \
        --linking-zarr linking.zarr
    """
}
