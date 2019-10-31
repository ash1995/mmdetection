from .coco import CocoDataset
from .registry import DATASETS

@DATASETS.register_module
class Xview2Dataset(CocoDataset):

    CLASSES = ('undamaged', 'minor-damage', 'major-damage', 'destroyed')
