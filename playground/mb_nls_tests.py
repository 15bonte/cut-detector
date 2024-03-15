""" Execution of the pipeline with the New Layer System
"""

import cut_detector.mb_blob.layers as nls
from cut_detector.mb_blob.savers import SaveLogger
from cut_detector.mb_blob.mb_blob_runner import simple_run

THREEFRAME_FP = "./src/cut_detector/data/mid_bodies_movies_test/example_video_mitosis_t28-30.tiff"
HARD_BIN_AOC_PIPELINE = [
    # nls.PlotImage(label="Raw")
    nls.OnFrame([0], nls.PlotImage(label="Raw")),
    nls.WriteTime(),
    nls.HardBinaryNormalizer(125),
    # nls.PlotImage(label="Post Bin")
    nls.OnFrame([1], nls.PlotImage(label="Post Bin")),
    nls.AreaOpeningNormalizer(50),
    nls.AreaClosingNormalizer(200),
    # nls.PlotImage(label="Post AO/AC")
    nls.OnFrame([2], nls.PlotImage(label="Post AO/AC")),
    nls.LapOfGauss(min_sig=3, max_sig=6, n_sig=10, threshold=0.1),
    nls.SaveStdBlobMetrics(),
    
]

file = THREEFRAME_FP
pipeline = HARD_BIN_AOC_PIPELINE

o = simple_run(
    pipeline=pipeline,
    tiff_path=file,
    savers=[SaveLogger(False)],
    pipeline_name="Prototype NLS Pipeline"
)