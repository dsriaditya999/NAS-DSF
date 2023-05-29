# coding:utf-8
import os
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from collections import namedtuple
from ipdb import set_trace as st
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
CityscapesClass = namedtuple(
    "CityscapesClass",
    ["name", "id", "train_id", "category", "category_id", "has_instances", "ignore_in_eval", "color"],
)

mf_classes = [
    CityscapesClass("unlabelled", 0, 0, "void", 0, False, True, (0, 0, 0)),
    CityscapesClass("car", 1, 1, "void", 0, False, True, (64, 0, 128)),
    CityscapesClass("person", 2, 2, "void", 0, False, True, (64, 64, 0)),
    CityscapesClass("bike", 3, 3, "void", 0, False, True, (0, 128, 192)),
    CityscapesClass("curve", 4, 4, "void", 0, False, True, (0, 0, 192)),
    CityscapesClass("car_stop", 5, 5, "void", 0, False, True, (128, 128, 0)),
    CityscapesClass("guardrail", 6, 6, "void", 0, False, True, (64, 64, 128)),
    CityscapesClass("color_cone", 7, 7, "void", 0, False, True, (192, 128, 128)),
    CityscapesClass("bump", 8, 8, "void", 0, False, True, (192, 64, 0))
]


class MF_dataset(Dataset):
    def __init__(self, data_dir, split, have_label, is_thermal=False, input_h=480, input_w=640 ,transform=[]):
        super(MF_dataset, self).__init__()

        assert split in ['train', 'val', 'test'], 'split must be "train"|"val"|"test"'

        with open(os.path.join(data_dir, split+'.txt'), 'r') as f:
            self.names = [name.strip() for name in f.readlines()]

        self.data_dir  = data_dir
        self.split     = split
        self.input_h   = input_h
        self.input_w   = input_w
        self.transform = A.Compose(transform)
        self.is_train  = have_label
        self.thermal = is_thermal
        self.n_data    = len(self.names)


    def read_image(self, name, folder):
        file_path = os.path.join(self.data_dir, '%s/%s.png' % (folder, name))
        image     = np.array(Image.open(file_path)) # (w,h,c)
        image.flags.writeable = True
        return image

    def get_train_item(self, index):
        name  = self.names[index]
        '''
        image = self.read_image(name, 'images')
        label = self.read_image(name, 'labels')

        for func in self.transform:
            image, label = func(image, label)

        image = np.array(Image.fromarray(image).resize((self.input_w, self.input_h)), dtype=np.float32).transpose((2,0,1))/255
        label = np.array(Image.fromarray(label).resize((self.input_w, self.input_h)), dtype=np.int64)
        
        return torch.tensor(image), torch.tensor(label)#, name
        '''
        img_path = os.path.join(self.data_dir, '%s/%s.png' % ('images', name))
        label_path = os.path.join(self.data_dir, '%s/%s.png' % ('labels', name))

        full_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = None
        if self.thermal:
            img = full_img[:, :, 3]
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
            #print(img.shape)
        else:
            img = full_img[:, :, :3]
            #print(img.shape)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        label = cv2.imread(label_path, -1).astype(np.int8)
        tform_data = self.transform(image=img, mask=label)
        img = tform_data['image']
        mask = tform_data['mask']
        img, mask = torch.from_numpy(img), torch.from_numpy(mask)
        img = img.permute(2,0,1)
        mask = mask.permute(0,1)
        mask[mask == 9] = -1
        img = img.type(torch.FloatTensor)
        mask = mask.type(torch.LongTensor)
        return img, mask

    def get_test_item(self, index):
        name  = self.names[index]
        image = self.read_image(name, 'images')
        image = np.array(Image.fromarray(image).resize((self.input_w, self.input_h)), dtype=np.float32).transpose((2,0,1))/255
        return torch.tensor(image)#, name


    def __getitem__(self, index):
        if self.is_train is True:
            return self.get_train_item(index)
        else: 
            return self.get_test_item (index)

    def __len__(self):
        return self.n_data

if __name__ == '__main__':
    data_dir = '../../ir_seg_dataset/'
    a = MF_dataset(data_dir, split='train', have_label=True, is_thermal=True)
    a.get_train_item(0)