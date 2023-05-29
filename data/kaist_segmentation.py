import glob
import os
from collections import namedtuple

import albumentations as A
import cv2
import numpy as np

import torch
import torch.utils.data as data
from albumentations.pytorch import ToTensorV2


class KAISTSegmentation(data.Dataset):
    
    CityscapesClass = namedtuple(
        "CityscapesClass",
        ["name", "id", "train_id", "category", "category_id", "has_instances", "ignore_in_eval", "color"],
    )

    classes = [
        CityscapesClass("unlabeled", 0, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("ego vehicle", 1, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("rectification border", 2, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("out of roi", 3, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("static", 4, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("dynamic", 5, 255, "void", 0, False, True, (111, 74, 0)),
        CityscapesClass("ground", 6, 255, "void", 0, False, True, (81, 0, 81)),
        CityscapesClass("road", 7, 0, "flat", 1, False, False, (128, 64, 128)),
        CityscapesClass("sidewalk", 8, 1, "flat", 1, False, False, (244, 35, 232)),
        CityscapesClass("parking", 9, 255, "flat", 1, False, True, (250, 170, 160)),
        CityscapesClass("rail track", 10, 255, "flat", 1, False, True, (230, 150, 140)),
        CityscapesClass("building", 11, 2, "construction", 2, False, False, (70, 70, 70)),
        CityscapesClass("wall", 12, 3, "construction", 2, False, False, (102, 102, 156)),
        CityscapesClass("fence", 13, 4, "construction", 2, False, False, (190, 153, 153)),
        CityscapesClass("guard rail", 14, 255, "construction", 2, False, True, (180, 165, 180)),
        CityscapesClass("bridge", 15, 255, "construction", 2, False, True, (150, 100, 100)),
        CityscapesClass("tunnel", 16, 255, "construction", 2, False, True, (150, 120, 90)),
        CityscapesClass("pole", 17, 5, "object", 3, False, False, (153, 153, 153)),
        CityscapesClass("polegroup", 18, 255, "object", 3, False, True, (153, 153, 153)),
        CityscapesClass("traffic light", 19, 6, "object", 3, False, False, (250, 170, 30)),
        CityscapesClass("traffic sign", 20, 7, "object", 3, False, False, (220, 220, 0)),
        CityscapesClass("vegetation", 21, 8, "nature", 4, False, False, (107, 142, 35)),
        CityscapesClass("terrain", 22, 9, "nature", 4, False, False, (152, 251, 152)),
        CityscapesClass("sky", 23, 10, "sky", 5, False, False, (70, 130, 180)),
        CityscapesClass("person", 24, 11, "human", 6, True, False, (220, 20, 60)),
        CityscapesClass("rider", 25, 12, "human", 6, True, False, (255, 0, 0)),
        CityscapesClass("car", 26, 13, "vehicle", 7, True, False, (0, 0, 142)),
        CityscapesClass("truck", 27, 14, "vehicle", 7, True, False, (0, 0, 70)),
        CityscapesClass("bus", 28, 15, "vehicle", 7, True, False, (0, 60, 100)),
        CityscapesClass("caravan", 29, 255, "vehicle", 7, True, True, (0, 0, 90)),
        CityscapesClass("trailer", 30, 255, "vehicle", 7, True, True, (0, 0, 110)),
        CityscapesClass("train", 31, 16, "vehicle", 7, True, False, (0, 80, 100)),
        CityscapesClass("motorcycle", 32, 17, "vehicle", 7, True, False, (0, 0, 230)),
        CityscapesClass("bicycle", 33, 18, "vehicle", 7, True, False, (119, 11, 32)),
        CityscapesClass("license plate", -1, -1, "vehicle", 7, False, True, (0, 0, 142)),
    ]


    def __init__(self, root, split, img_size=768, is_thermal = False):
        self.root = root
        
        folder = None
        if split == 'val':
            folder = 'val_night'
        else:
            folder = 'val_day'    

        annotation_files = sorted(glob.glob(os.path.join(root, 'segmentation', 'labels', folder, '*.png')))

        self.data = []
        for file in annotation_files:
            set_no, seq_no, name = os.path.basename(file).split('_')

            img_path = os.path.join(root, 'rgbt-ped-detection', 'data', 'kaist-cvpr15', 'images', set_no, seq_no, 'visible', name)

            if is_thermal:
                img_path = os.path.join(root, 'rgbt-ped-detection', 'data', 'kaist-cvpr15', 'images', set_no, seq_no, 'lwir', name)
            
            assert os.path.exists(img_path), img_path
            entry = {
                'img_path' : img_path,
                'label_path' : file
            } 
            self.data.append(entry)

        
        if split == 'val':
            self.transforms = A.Compose([
                A.LongestMaxSize(img_size),
                A.Normalize(),
                ToTensorV2(),
            ])
        else:
            self.transforms = A.Compose([
                A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.5, 2.0)),
                A.HorizontalFlip(p=0.5),
                A.Normalize(),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        entry = self.data[index] 
        img_path = entry['img_path']
        label_path = entry['label_path']

        img = cv2.imread(img_path, 1)
        label = cv2.imread(label_path, -1).astype(np.int8)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        tform_data = self.transforms(image=img, mask=label)
        img = tform_data['image']
        mask = tform_data['mask']
        mask[mask == 19] = -1

        mask = mask.type(torch.LongTensor)
        return img, mask







# root = '/home/carson/data/cityscapes/data/cityscapes'
# def create_cityscapes_dataloaders(root, batch_size, num_workers):

#     val_transforms = Compose([
#         Resize(size=512, max_size=768),
#         ToTensor(),
#     ])

#     val_dataset = CityScapesConverted(root, split='val', mode='fine', target_type='semantic', transforms=val_transforms)

#     train_transforms = Compose([
#         RandomResizedCrop(size=(768, 768), scale=(0.5, 2.0)),
#         RandomHorizontalFlip(),
#         ToTensor(),
#     ])
#     fine_train_dataset = CityScapesConverted(root, split='train', mode='fine', target_type='semantic', transforms=train_transforms)
#     coarse_train_extra_dataset = CityScapesConverted(root, split='train_extra', mode='coarse', target_type='semantic', transforms=train_transforms)
#     train_dataset = data.ConcatDataset([fine_train_dataset, coarse_train_extra_dataset])
    
#     val_loader = data.DataLoader(
#         val_dataset,
#         batch_size=batch_size, 
#         num_workers=num_workers, 
#         shuffle=False, 
#     )

#     train_loader = data.DataLoader(
#         train_dataset,
#         batch_size=batch_size, 
#         num_workers=num_workers, 
#         shuffle=True, 
#     )

#     return train_loader, val_loader

if __name__=="__main__":

    dataset = KAISTSegmentation('/home/carson/data/kaist/', 'val')
    for i in range(len(dataset)):
        x = dataset.__getitem__(i)
        print(x[0])
        print(np.unique(x[1].numpy()))
        print(x[0].shape, x[1].shape)

        break

    print(len(dataset))

    dataset = KAISTSegmentation('/home/carson/data/kaist/', 'train')
    for i in range(len(dataset)):
        x = dataset.__getitem__(i)
        print(x[0])
        print(np.unique(x[1].numpy()))
        print(x[0].shape, x[1].shape)

        break

    print(len(dataset))