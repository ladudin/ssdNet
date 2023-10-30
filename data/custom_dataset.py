import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import cv2
import json


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict_file, transform=None, add_path=None):
        self.transform = transform

        with open(data_dict_file, 'r') as f:
            self.data_dict = json.load(f)
        if add_path:
            self.data_dict = {add_path + i:j for i,j in self.data_dict.items()}

        self.imgs = list(self.data_dict.keys())


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target).
                   target is the object returned by ``coco.loadAnns``.
        """
        im, gt, h, w = self.pull_item(index)
        return im, gt
       
    
    def pull_item(self, index):
        img_path = self.imgs[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        height, width = img.shape[:2]
        
        num_objs = len(self.data_dict[img_path])
        boxes = []
        for i in range(num_objs):
            bbox = self.data_dict[img_path][i]
            xmin = bbox[0] * width
            xmax = bbox[1] * width
            ymin = bbox[2] * height
            ymax = bbox[3] * height
            boxes.append([xmin, ymin, xmax, ymax])
        
        boxes = np.asarray(boxes)  # N x 4
        labels = np.zeros((len(boxes), 1))  # N x 1
        target = np.concatenate((boxes, labels), axis=1)  # N x 5

        # if self.target_transform is not None:
        #     target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4],
                                                target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]

            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width

    def __len__(self):
        return len(self.imgs)

    # def image_aspect_ratio(self, image_index):
    #     img_path = self.imgs[image_index]
    #     img = Image.open(img_path).convert("RGB") # slow
    #     w, h = img.size
    #     return w / h

    def num_classes(self):
        return 2
