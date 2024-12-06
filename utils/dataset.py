# -*- coding: utf-8 -*-

import torch.utils.data as data
import os

import torch
import numpy as np
import cv2

class MyDataset(data.Dataset):
    def __init__(self, imgpath, mskpath, resize_h=512, resize_w=512):

        imgs = []
        img_list = os.listdir(imgpath)
        img_list = sorted(img_list)
        label_list = os.listdir(mskpath)
        label_list = sorted(label_list)

        for i in range(len(img_list)):
            img = img_list[i]
            mask = label_list[i]
            imgs.append(["{}/{}".format(imgpath, img), "{}/{}".format(mskpath, mask)])

        self.imgs = imgs
        self.resize_h = resize_h
        self.resize_w = resize_w

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]

        img_x = cv2.imread(x_path)

        img_x = cv2.resize(img_x, (self.resize_w, self.resize_h))
        img_x = np.transpose(img_x, axes=(2, 0, 1))
        img_x = torch.from_numpy(img_x).type(torch.FloatTensor)

        img_y = cv2.imread(y_path)

        img_y = cv2.resize(img_y, (self.resize_w, self.resize_h))
        img_y_b, img_y_g, img_y_r = cv2.split(img_y)
        img_y = torch.from_numpy(img_y_b).type(torch.LongTensor)

        return img_x, img_y

    def __len__(self):
        return len(self.imgs)

