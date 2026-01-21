import os
from yacs.config import CfgNode as CN

_C = CN()

_C.MODE = "semantic"

# Dataset configuration
_C.DATASET = CN()
_C.DATASET.SEMANTIC_MASK_SUBDIR  = "semantic_masks"
_C.DATASET.SUBSET_LIST = "image_list.txt"

# Semantic segmentation: class configuration
_C.SEGMENTATION = CN()
_C.SEGMENTATION.IDS = [1, 2, 3, 4]
_C.SEGMENTATION.IGNORE_ID = 0
_C.SEGMENTATION.NAMES = ['static_obstacle', 'dynamic_obstacle', 'water', 'sky']
_C.SEGMENTATION.STATIC_OBSTACLE_CLASS = 1
_C.SEGMENTATION.DYNAMIC_OBSTACLE_CLASS = 2
_C.SEGMENTATION.WATER_CLASS = 3
_C.SEGMENTATION.SKY_CLASS = 4
_C.SEGMENTATION.COLORS = [[0, 255,  0],  # Static obstacles RGB color
                           [255, 0,  0], # Dynamic obstacles RGB color
                          [ 0, 0, 255],  # Water RGB color
                          [148, 0, 211]]  # Sky RGB color

# All Paths
_C.PATHS = CN()
_C.PATHS.RESULTS       = "./results"                # Path to where the results will be saved
_C.PATHS.DATASET_ROOT  = "/path/to/dataset"         # Path to where the dataset is stored
_C.PATHS.PREDICTIONS   = "/path/to/predictions/"    # Path to where the segmentation predictions are stored

# Evaluation configuration
_C.EVALUATION = CN()

# Progress output configuration
_C.PROGRESS = CN()
_C.PROGRESS.METRICS = ['mIoU']

def get_cfg(config_file=None):
    cfg = _C.clone()
    if config_file is not None:
       cfg.merge_from_file(config_file)
    cfg.freeze()
    return cfg
