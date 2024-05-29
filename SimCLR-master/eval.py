import torch
import sys
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
import yaml
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from models.resnet_simclr import ResNetSimCLR
import importlib.util
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import train_test_split
from dataset_eval import HPDataset
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


def eval(model,data_root,device,config):
    train_loader,test_loader = load_dataset(data_root,config)
    X_train_feature = []
    y_train = []
    for (batch_x, _, batch_x_neg, _), batch_y in train_loader:
        batch_x = batch_x.to(device)
        features, _ = model(batch_x)
        X_train_feature.extend(features.cpu().detach().numpy())
        y_train.extend(batch_y.cpu().detach().numpy())
    X_train_feature = np.array(X_train_feature)
    X_test_feature = []
    y_train = np.array(y_train)
    y_test = []
    i=0
    for (batch_x, _, batch_x_neg, _), batch_y in test_loader:
        batch_x = batch_x.to(device)
        features, _ = model(batch_x)
        X_test_feature.extend(features.cpu().detach().numpy())
        y_test.extend(batch_y.cpu().detach().numpy())
    X_test_feature = np.array(X_test_feature)
    y_test = np.array(y_test)
    scaler = preprocessing.StandardScaler()
    print('ok')
    scaler.fit(X_train_feature)
    #print(X_test_feature.shape)
    #print(y_test.shape)
    linear_model_eval(scaler.transform(X_train_feature), y_train, scaler.transform(X_test_feature), y_test)



def load_dataset(root ,config):
    train_dataset = pd.read_excel('../HPyloriData/HP_WSI-CoordAnnotatedWindows.xlsx')
    train_dataset = train_dataset[train_dataset['Deleted']==0][train_dataset['Cropped']==0][train_dataset['Presence']!=0].reset_index()
    # gss = GroupShuffleSplit(n_splits=1, test_size=0.2)
    train_loader,valid_loader = train_test_split(train_dataset,test_size=0.3,stratify=train_dataset['Presence'])
    # X = np.array(train_dataset['index'])
    # y = np.array(train_dataset['Presence'])
    # groups = np.array(train_dataset['Pat_ID'])
    # splits = gss.split(X, y, groups)
    # for train_index, test_index in splits:
    #     X_train, X_test = X[train_index], X[test_index]
    #     print(train_dataset)
    train_loader = train_loader.reset_index().drop('level_0',axis=1)
    valid_loader = valid_loader.reset_index().drop('level_0',axis=1)

    train_data_path = '../HPyloriData/annotated_windows'
    train_set = HPDataset(train_loader,path=train_data_path,train=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((32,32)),
                                                                                                        transforms.Normalize([0.8061, 0.8200, 0.8886], [0.0750, 0.0563, 0.0371])]))
    valid_set = HPDataset(valid_loader,path=train_data_path,train=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((32,32)),
                                                                                                        transforms.Normalize([0.8061, 0.8200, 0.8886], [0.0750, 0.0563, 0.0371])]))

    train_dl = DataLoader(train_set,batch_size=config['batch_size'],shuffle=True,drop_last=True)
    test_dl = DataLoader(valid_set,batch_size=config['batch_size'],shuffle=True,drop_last=True)
    return train_dl, test_dl

def load_model(checkpoints_folder,device):
    model =ResNetSimCLR(**config['model'])
    model.eval()
    state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'), map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model = model.to(device)
    return model

    
def linear_model_eval(X_train, y_train, X_test, y_test):
    
    clf = LogisticRegression(random_state=0, max_iter=10000, solver='lbfgs', C=1.0)
    clf.fit(X_train, y_train)
    print("Logistic Regression feature eval")
    print("Train score:", clf.score(X_train, y_train))
    print("Test score:", clf.score(X_test, y_test))

    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    # Plot the confusion matrix
    disp.plot()

    # Save the figure
    plt.savefig('confusion_matrix.png')

    # y_pred_proba = clf.predict_proba(X_test)[:, 1]
    

    # fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    # plt.plot(fpr, tpr)
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.savefig('roc_curve.png')
    # plt.show()

 
    print("-------------------------------")
    neigh = KNeighborsClassifier(n_neighbors=10)
    neigh.fit(X_train, y_train)
    print("KNN feature eval")
    print("Train score:", neigh.score(X_train, y_train))
    print("Test score:", neigh.score(X_test, y_test))

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)
    folder_name = './runs/May17_20-55-07_userme'
    checkpoints_folder = os.path.join(folder_name, 'checkpoints')
    config = yaml.load(open(os.path.join(checkpoints_folder, "config.yaml"), "r"))
    data_root = './data/'
    model = load_model(checkpoints_folder,device)
    eval(model,data_root,device,config)
    
    #load_dataset()