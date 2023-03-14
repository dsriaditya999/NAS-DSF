""" Dataset factory
"""
import os
from collections import OrderedDict
from pathlib import Path

from effdet.data.parsers import *
from effdet.data.parsers import create_parser

from .dataset_config import *
from .dataset import FusionDataset

def create_dataset(name, root, splits=('train', 'val')):
    if isinstance(splits, str):
        splits = (splits,)
    name = name.lower()
    root = Path(root)
    dataset_cls = FusionDataset
    datasets = OrderedDict()
    if name == 'flir_aligned':
        dataset_cfg = FlirAlignedCfg()
        for s in splits:
            if s not in dataset_cfg.splits:
                raise RuntimeError(f'{s} split not found in config')
            split_cfg = dataset_cfg.splits[s]
            ann_file = root / split_cfg['ann_filename']
            parser_cfg = CocoParserCfg(
                ann_filename=ann_file,
                has_labels=split_cfg['has_labels']
            )
            datasets[s] = dataset_cls(
                thermal_data_dir=root / Path(split_cfg['img_dir']),
                rgb_data_dir=root / Path(split_cfg['img_dir'].replace('thermal', 'rgb')),
                parser=create_parser(dataset_cfg.parser, cfg=parser_cfg),
            )
    elif name == 'llvip':
        dataset_cfg = LLVIPCfg()
        for s in splits:
            if s not in dataset_cfg.splits:
                raise RuntimeError(f'{s} split not found in config')
            split_cfg = dataset_cfg.splits[s]
            ann_file = root / split_cfg['ann_filename']
            print(ann_file)
            parser_cfg = CocoParserCfg(
                ann_filename=ann_file,
                has_labels=split_cfg['has_labels']
            )
            datasets[s] = dataset_cls(
                thermal_data_dir=root / Path(split_cfg['img_dir']),
                rgb_data_dir=root / Path(split_cfg['img_dir'].replace('infrared', 'visible')),
                parser=create_parser(dataset_cfg.parser, cfg=parser_cfg),
            )
    else:
        assert False, f'Unknown dataset parser ({name})'

    datasets = list(datasets.values())
    return datasets if len(datasets) > 1 else datasets[0]
