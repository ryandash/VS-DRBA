# VapourSynth .vpy example: VS-DRBA RIFE 4.25, 1080p -> 48 fps
#
# Edit REPO_ROOT, INPUT_FILE, and (if needed) the source plugin below.
# Open this file with the VapourSynth editor or render it with vspipe.

import os
import sys
import vapoursynth as vs

core = vs.core

# Path to the cloned VS-DRBA repository (the checkout containing the "vsdrba" folder).
REPO_ROOT = r"C:\VideoTools\VS-DRBA"
INPUT_FILE = r"D:\Videos\input_1080p.mkv"
TRT_CACHE = r"D:\VideoTools\VS-DRBA-TRT-Cache"

if not os.path.isdir(os.path.join(REPO_ROOT, "vsdrba")):
    raise RuntimeError(
        "REPO_ROOT must point to your VS-DRBA checkout, with a 'vsdrba' subfolder."
    )
if not os.path.isfile(INPUT_FILE):
    raise RuntimeError(f"Input file does not exist: {INPUT_FILE}")

# Put this checkout first so Python imports the branch instead of an older pip install.
sys.path.insert(0, REPO_ROOT)

from vsdrba import drba_rife

# Example source using L-SMASH Works. If you use another source plugin,
# replace this line with your usual source filter.
clip = core.lsmas.LWLibavSource(INPUT_FILE)

# DRBA RIFE accepts RGBH (FP16) or RGBS (FP32). RGBH is recommended here.
clip = clip.resize.Bicubic(
    format=vs.RGBH,
    matrix_in_s="709",
)

clip = drba_rife(
    clip,
    model="4.25",
    factor_num=2,
    factor_den=1,
    sc=True,
    sc_threshold=0.15,
    trt=True,
    trt_static_shape=True,
    trt_optimization_level=5,
    trt_max_aux_streams=None,
    block0_batch_size=2,  # set to 1 to benchmark the original two-call path
    trt_cache_dir=TRT_CACHE,
)

clip.set_output()
