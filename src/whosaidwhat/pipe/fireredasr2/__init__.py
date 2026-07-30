# Copyright 2026 Xiaohongshu. (Author: Kaituo Xu)

import os
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

__version__ = "0.0.1"

from .asr import FireRedAsr2, FireRedAsr2Config

__all__ = [
    "__version__",
    "FireRedAsr2",
    "FireRedAsr2Config",
]
