from PIL import Image
import random
import os
import numpy as np
import torchvision
import copy

class DiagnosisDataset():
    def __init__(self, patient, path, transform=None):
        self.patient=patient
        self.path=path
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    def __getitem__(self, item):
        list_of_imgs = []
        for filename in os.listdir(self.path +'/'+ self.patient):
             print(filename)
             list_of_imgs.append(Image.open(self.path +'/'+ self.patient + '/' + filename).convert('RGB'))
        
        if self.transform!=None:
                for i in len(list_of_imgs):
                    list_of_imgs[i] = self.transform(list_of_imgs[i])

        # print(anchor_img.shape)
        return list_of_imgs