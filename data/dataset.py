""" Fusion dataset for detection
"""
import torch.utils.data as data
import numpy as np
import copy
import torch

from PIL import Image
from effdet.data.parsers import create_parser


class OnlineTrainingDataset:
    def __init__(self, buffer_size):
        super(OnlineTrainingDataset, self).__init__()
        self.training_sample_buffer = [] # Implement as bounded queue
        self.buffer_size = buffer_size

    def get_training_data(self):
        for i, batch in enumerate(self.training_sample_buffer):
            thermal_img, rgb_img, target = batch[0], batch[1], batch[2]
            if i == 0:
                thermal_img_train = thermal_img
                rgb_img_train = rgb_img
                target_train = copy.deepcopy(target)
            else:
                thermal_img_train = torch.cat((thermal_img_train, thermal_img), dim=0)
                rgb_img_train = torch.cat((rgb_img_train, rgb_img), dim=0)
                for k, v in target_train.items():
                    target_train[k] = torch.cat((target_train[k], target[k]), dim=0)

        return thermal_img_train, rgb_img_train, target_train

    def len(self):
        return len(self.training_sample_buffer)

    def add_current_data(self, thermal_img, rgb_img, target):
        if self.len() == self.buffer_size:
            self.training_sample_buffer.pop(0)
        self.training_sample_buffer.append([thermal_img, rgb_img, target])


class FusionDataset(data.Dataset):
    """ Fusion Dataset for Object Detection. Use with parsers for COCO, VOC, and OpenImages.
    Args:
        parser (string, Parser):
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
    """
    def __init__(self, thermal_data_dir, rgb_data_dir, parser=None, parser_kwargs=None, transform=None):
        super(FusionDataset, self).__init__()
        parser_kwargs = parser_kwargs or {}
        self.thermal_data_dir = thermal_data_dir
        self.rgb_data_dir = rgb_data_dir
        if isinstance(parser, str):
            self._parser = create_parser(parser, **parser_kwargs)
        else:
            assert parser is not None and len(parser.img_ids)
            self._parser = parser
        self._transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (thermal_image, rgb_image, annotations (target)).
        """
        img_info = self._parser.img_infos[index]
        target = dict(img_idx=index, img_size=(img_info['width'], img_info['height']))
        if self._parser.has_labels:
            ann = self._parser.get_ann_info(index)
            target.update(ann)

        thermal_img_path = self.thermal_data_dir / img_info['file_name']
        thermal_img = Image.open(thermal_img_path).convert('RGB')
        rgb_img_path = self.rgb_data_dir / img_info['file_name'].replace('PreviewData', 'RGB')
        rgb_img = Image.open(rgb_img_path).convert('RGB')
        if self.transform is not None:
            thermal_img, rgb_img, target = self.transform(thermal_img, rgb_img, target)

        return thermal_img, rgb_img, target

    def __len__(self):
        return len(self._parser.img_ids)

    @property
    def parser(self):
        return self._parser

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, t):
        self._transform = t
