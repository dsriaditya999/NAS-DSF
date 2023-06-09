import os
from dataclasses import dataclass, field
from typing import Dict

from effdet.data.dataset_config import *


@dataclass
class FlirAlignedCfg(CocoCfg):
    variant: str = ''
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(ann_filename='images_thermal_train_night/night_flir.json', img_dir='images_thermal_train_night/data/', has_labels=True),
        val=dict(ann_filename='images_thermal_val_night/night_flir.json', img_dir='images_thermal_val_night/data/', has_labels=True),
        test=dict(ann_filename='images_thermal_val_night/night_flir.json', img_dir='images_thermal_val_night/data/', has_labels=True),
    ))


@dataclass
class LLVIPCfg(CocoCfg):
    variant: str = ''
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(ann_filename='llvip_train.json', img_dir='infrared/train/', has_labels=True),
        val=dict(ann_filename='llvip_test.json', img_dir='infrared/test/', has_labels=True),
        test=dict(ann_filename='llvip_test.json', img_dir='infrared/test/', has_labels=True),
    ))