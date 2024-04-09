""" Execution of the pipeline with the New Layer System
"""

import layers as nls
from savers import SaveLogger
from mb_blob_runner import simple_run, run_test_environment

MB_MV_TEST = "./src/cut_detector/data/mid_bodies"
MB_MV_PPLN_TEST = "./src/cut_detector/data/mid_bodies_movies_test_ppln" 
SAVE_DIR = MB_MV_PPLN_TEST

THREEFRAME_FP = "./src/cut_detector/data/mid_bodies_movies_test/example_video_mitosis_t28-30.tiff"
LUCI_A_FP = "./src/cut_detector/data/mid_bodies_movies_test/a_siLuci-1_mitosis_33_7_to_63.tiff"
CEP_2 = "./src/cut_detector/data/mid_bodies_movies_test/cep2_20231019-t1_siCep55-50-4_mitosis_24_17_to_104.tiff"

HARD_BIN_AOC_PIPELINE = [
    nls.WriteImg(as_key="src"),
    nls.OnFrame([0], nls.PlotImage(label="Raw")),
    nls.WriteTime(),
    nls.HardBinaryNormalizer(125),
    nls.OnFrame([0], nls.PlotImage(label="Post Bin")),
    nls.AreaOpeningNormalizer(50),
    nls.AreaClosingNormalizer(200),
    nls.OnFrame([0], nls.PlotImage(label="Post AO/AC")),
    nls.LapOfGauss(min_sig=3, max_sig=6, n_sig=10, threshold=0.1),
    nls.SaveStdBlobMetrics(),
    nls.OnFrame([0], nls.PlotBlobs())
]

MINMAX_PIPELINE = [
    nls.WriteImg(as_key="src"),
    nls.WriteTime(),
    nls.MinMaxNormalizer(),
    nls.OnFrame(
        [0], 
        nls.LapOfGauss(5, 10, 5, 0.2, nls.BlobLogVisuSettings(0.2, [0, 4])),
        nls.LapOfGauss(5, 10, 5, 0.2)
    ),
    nls.OnFrame([0], nls.PlotBlobs()),
    nls.SaveStdBlobMetrics(),
]

BETTER_MINMAX_PIPELINE = [
    nls.WriteImg(as_key="src"),
    nls.WriteTime(),
    nls.MinMaxNormalizer(),
    nls.OnFrame(
        [16], 
        nls.Sequence(
            nls.LapOfGauss(5, 10, 5, 0.1, nls.BlobLogVisuSettings(0.01, [0, 4])),
            # nls.PlotBlobs(plt_saver=nls.PlotSaver(filedir=SAVE_DIR, filename="blobs_seen")),
        ),
        nls.Sequence(
            nls.LapOfGauss(5, 10, 5, 0.1),
            # nls.PlotBlobs(plt_saver=nls.PlotSaver(filedir=SAVE_DIR, filename="blobs_seen"), should_show=False),
        )
    ),
    nls.SaveStdBlobMetrics(),
]

DEMAKE_MINMAX_PIPELINE = [
    nls.WriteImg(as_key="src"),
    nls.WriteTime(),
    nls.MinMaxNormalizer(),
    nls.LapOfGauss(5, 10, 5, 0.1),
    nls.PlotBlobs(plt_saver=nls.PlotSaver(filedir=SAVE_DIR, filename="blobs_seen"), should_show=False),
    nls.SaveStdBlobMetrics(),
]

DEMAKE_MINMAX_WIDER_PIPELINE = [
    nls.WriteImg(as_key="src"),
    nls.WriteTime(),
    nls.MinMaxNormalizer(),
    nls.LapOfGauss(2, 10, 5, 0.1),
    nls.PlotBlobs(plt_saver=nls.PlotSaver(filedir=SAVE_DIR, filename="blobs_seen"), should_show=False),
    nls.SaveStdBlobMetrics(),
]

DEMAKE_MINMAX_STR02_PIPELINE = [
    nls.WriteImg(as_key="src"),
    nls.WriteTime(),
    nls.MinMaxNormalizer(),
    nls.LapOfGauss(2, 10, 5, 0.2),
    nls.PlotBlobs(plt_saver=nls.PlotSaver(filedir=SAVE_DIR, filename="blobs_seen"), should_show=False),
    nls.SaveStdBlobMetrics(),
]

DEMAKE_MINMAX_STR03_PIPELINE = [
    nls.WriteImg(as_key="src"),
    nls.WriteTime(),
    nls.MinMaxNormalizer(),
    nls.LapOfGauss(2, 10, 5, 0.2),
    nls.PlotBlobs(plt_saver=nls.PlotSaver(filedir=SAVE_DIR, filename="blobs_seen"), should_show=False),
    nls.SaveStdBlobMetrics(),
]

DEMAKE_MINPERC_PIPELINE = [
    nls.WriteImg(as_key="src"),
    nls.WriteTime(),
    nls.MinPercentileNormalizer(percentile=99),
    nls.LapOfGauss(5, 10, 5, 0.1,),
    nls.PlotBlobs(plt_saver=nls.PlotSaver(filedir=SAVE_DIR, filename="blobs_seen"), should_show=False),
    nls.SaveStdBlobMetrics(),
]

DEMAKE_MINPERC_HARDER_PIPELINE = [
    nls.WriteImg(as_key="src"),
    nls.WriteTime(),
    nls.MinPercentileNormalizer(percentile=99),
    nls.LapOfGauss(5, 10, 3, 0.3),
    nls.PlotBlobs(plt_saver=nls.PlotSaver(filedir=SAVE_DIR, filename="blobs_seen"), should_show=False),
    nls.SaveStdBlobMetrics(),
]

PROTO_DOG_PIPELINE = [
    nls.WriteImg(as_key="src"),
    nls.WriteTime(),
    nls.MinMaxNormalizer(),
    nls.DiffOfGauss(5, 10),
    nls.PlotBlobs(plt_saver=nls.PlotSaver(filedir=SAVE_DIR, filename="blobs_seen"), should_show=False),
    nls.SaveStdBlobMetrics(),
]

PROTO_DOH_PIPELINE = [
    nls.WriteImg(as_key="src"),
    nls.WriteTime(),
    nls.MinMaxNormalizer(),
    nls.DetOfHess(5, 10, 5, 0.1),
    nls.PlotBlobs(plt_saver=nls.PlotSaver(filedir=SAVE_DIR, filename="blobs_seen"), should_show=False),
    nls.SaveStdBlobMetrics(),
]



file = CEP_2
pipeline = BETTER_MINMAX_PIPELINE


# o = simple_run(
#     pipeline=pipeline,
#     tiff_path=file,
#     savers=[SaveLogger(False)],
#     pipeline_name="Prototype NLS Pipeline"
# )


# o = run_test_environment(
#     [THREEFRAME_FP],
#     {
#         "demake": DEMAKE_MINMAX_PIPELINE,
#         "p-dog": PROTO_DOG_PIPELINE,
#         "p-doh": PROTO_DOH_PIPELINE,
#     },
#     savers=[SaveLogger(False)],
# )

def run_example():
    o = run_test_environment(
        [CEP_2],
        {
            "minmax": DEMAKE_MINMAX_PIPELINE,
            "minmax_wider": DEMAKE_MINMAX_WIDER_PIPELINE,
            "minmax_wider_stro2": DEMAKE_MINMAX_STR02_PIPELINE,
            "minmax_wider_stro3": DEMAKE_MINMAX_STR03_PIPELINE,
            "minmax_wider": DEMAKE_MINMAX_WIDER_PIPELINE,
            "minperc": DEMAKE_MINPERC_PIPELINE,
            "minperc_strong": DEMAKE_MINPERC_HARDER_PIPELINE
        },
        savers=[SaveLogger(False)],
    )

if __name__ == "__main__":
    run_example()


