from PIL import Image
import random
import os
import numpy as np
import torchvision
import copy

class HPDataset():
    def __init__(self, df, path, train=True, transform=None):
        self.data_csv = df
        self.is_train = train
        self.transform = transform
        self.path = path
        if self.is_train:
            self.images = (df.iloc[:, 1].values) + "_" + (list(map(str, df.iloc[:, 2]))) + "." + [num.zfill(5) for num in list(map(str, df.iloc[:, 3]))]
            self.labels = df.iloc[:, 8].values
            self.index = df.index.values 

    
    def __len__(self):
        return len(self.images)
    def __getitem__(self, item):
        anchor_image_folder, anchor_image_name = self.images[item].split('.')
        anchor_image_path = self.path + '/' + anchor_image_folder + '/' + anchor_image_name + '.png'
        ###### Anchor Image #######
        anchor_img = Image.open(anchor_image_path).convert('RGB')
        anchor_label = self.labels[item]
        if self.transform!=None:
            anchor_img = self.transform(anchor_img)
        label = (anchor_label + 1)/2

        # print(anchor_img.shape)
        return anchor_img, label