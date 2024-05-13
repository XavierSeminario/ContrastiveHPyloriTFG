import os
import torch
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import json
from torchvision.transforms import transforms
from sklearn.model_selection import train_test_split
from dataset import TripletDataset
import random
from losses import *
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from model import ResNet_Triplet
from tqdm import tqdm
from utils import *
from sklearn.model_selection import GroupShuffleSplit


train_data_path = 'HPyloriData/annotated_windows'
train_data = pd.read_excel('./HPyloriData/HP_WSI-CoordAnnotatedWindows.xlsx')
train_data = train_data[train_data['Deleted']==0][train_data['Cropped']==0][train_data['Presence']!=0].reset_index()
print(train_data)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
X = np.array(train_data['index'])
y = np.array(train_data['Presence'])
groups = np.array(train_data['Pat_ID'])
splits = gss.split(X, y, groups)
for train_index, test_index in splits:
    X_train, X_test = X[train_index], X[test_index]
    # print(train_data)
    train_loader = train_data[np.isin(X, X_train)].reset_index().drop('level_0',axis=1)
    valid_loader = train_data[np.isin(X, X_test)].reset_index().drop('level_0',axis=1)

def get_train_dataset(IMAGE_SIZE=256):
    train_dataset = TripletDataset(train_loader,path=train_data_path,train=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((IMAGE_SIZE,IMAGE_SIZE))]))
    # test_dataset = TripletDataset(test_set,path=train_data_path,train=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((IMAGE_SIZE,IMAGE_SIZE))]))    
    return train_dataset

def get_test_dataset(IMAGE_SIZE=256):
    test_dataset = TripletDataset(valid_loader,path=train_data_path,train=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((IMAGE_SIZE,IMAGE_SIZE))]))    
    return test_dataset

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

IMAGE_SIZE = 28
BATCH_SIZE = 64
TEST_B_SIZE = 100
DEVICE = get_device()
LEARNING_RATE = 0.001
EPOCHS = 75

train_dataset = get_train_dataset(IMAGE_SIZE = IMAGE_SIZE)
train_dl = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
testr_dataset = get_test_dataset(IMAGE_SIZE = IMAGE_SIZE)
test_dl = DataLoader(testr_dataset,batch_size=TEST_B_SIZE,shuffle=True)

# print(train_data)
ResNet = ResNet_Triplet()
ResNet = ResNet.to(DEVICE)
optimizer = torch.optim.Adam(ResNet.parameters(),lr = LEARNING_RATE)
criterion = TripletLoss(2)
losses_train = []
losses_val =  []


for epoch in tqdm(range(EPOCHS), desc='Epochs'):
    ResNet.train()
    loss_acum = []
    for step, (anchor_img, positive_img, negative_img, anchor_label) in enumerate(tqdm(train_dl, desc='Training', leave=False)):
        if step == 4:
            test_anchor_img = anchor_img
            test_positive_img = positive_img
            test_negative_img = negative_img
            continue
        anchor_img = rotate_some(anchor_img,p=0.5).to(DEVICE)
        positive_img = rotate_some(positive_img,p=0.5).to(DEVICE)
        negative_img = rotate_some(negative_img,p=0.5).to(DEVICE)
        optimizer.zero_grad()
        anchor_out = ResNet(anchor_img)
        positive_out = ResNet(positive_img)
        negative_out = ResNet(negative_img)
        loss = criterion(anchor_out, positive_out, negative_out)
        loss.backward()
        optimizer.step()
        loss_acum.append(loss.cpu().detach().numpy())
    losses_train.append(np.mean(loss_acum))
    ResNet.eval()
    test_anchor_img = test_anchor_img.to(DEVICE)
    test_positive_img = test_positive_img.to(DEVICE)
    test_negative_img = test_negative_img.to(DEVICE)
    anchor_out = ResNet(test_anchor_img)
    positive_out = ResNet(test_positive_img)
    negative_out = ResNet(test_negative_img)
    loss = criterion(anchor_out, positive_out, negative_out)
    losses_val.append(np.mean(loss.cpu().detach().numpy()))
    print('Epoch: {}/{} — Loss: {:.4f}\n'.format(epoch+1, EPOCHS, np.mean(loss_acum)))

embeddings = ResNet.Feature_Extractor(anchor_img)

print("DONE")

# print(anchor_img.shape)
for step, (anchor_img, positive_img, negative_img, anchor_label) in enumerate(tqdm(test_dl, desc='Testing', leave=False)):
        if step == 1:
            anchor_img = anchor_img.to(DEVICE)
            break
embeddings = ResNet.Feature_Extractor(anchor_img)
tsne = TSNE(n_components=2, random_state=42, perplexity=20)
embeddings_2d = tsne.fit_transform(embeddings.detach().cpu().numpy())

plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=anchor_label.detach().tolist(), cmap='viridis', marker='o')
plt.title('Visualització t-SNE dels Embedings 64-Dimensionals')
plt.colorbar()
plt.show()
print("DONE FINAL")