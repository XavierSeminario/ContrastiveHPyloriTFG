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
        if self.is_train:
            anchor_label = self.labels[item]
            positive_list = self.index[self.index!=item][self.labels[self.index!=item]==anchor_label]

            positive_item = random.choice(positive_list)
            positive_image_folder, positive_image_name = self.images[positive_item].split('.')
            positive_image_path = self.path + '/' + positive_image_folder + '/' + positive_image_name + '.png'
            positive_img = Image.open(positive_image_path).convert('RGB')
            # positive_img = self.images[positive_item].reshape(28, 28, 1)

            negative_list = self.index[self.index!=item][self.labels[self.index!=item]!=anchor_label]
            negative_item = random.choice(negative_list)
            negative_negative_folder, negative_image_name = self.images[negative_item].split('.')
            negative_image_path = self.path + '/' + negative_negative_folder + '/' + negative_image_name + '.png'
            negative_img = Image.open(negative_image_path).convert('RGB')

            negative_item2 = random.choice(negative_list)
            negative_negative_folder2, negative_image_name2 = self.images[negative_item2].split('.')
            negative_image_path2 = self.path + '/' + negative_negative_folder2 + '/' + negative_image_name2 + '.png'
            negative_img2 = Image.open(negative_image_path2).convert('RGB')
            #negative_img = self.images[negative_item].reshape(28, 28, 1)
            if self.transform!=None:
                 anchor_img = self.transform(anchor_img)
                 positive_img = self.transform(positive_img)                   
                 negative_img = self.transform(negative_img)
                 negative_img2 = self.transform(negative_img2)
        angles = [90,180,270]
        positive_img=torchvision.transforms.functional.rotate(positive_img, angle=random.choice(angles))
        negative_img2  = torchvision.transforms.functional.rotate(negative_img2, angle=random.choice(angles))
        label = (anchor_label + 1)/2

        # print(anchor_img.shape)
        return (anchor_img, positive_img, negative_img, negative_img2), label