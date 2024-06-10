import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from dataset_no_cl import HPDataset
import torchvision.transforms as transforms


# Define the small neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Binary classification output
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x
    

class ResNet_Triplet(nn.Module):
    def __init__(self):
        super().__init__()
        self.Feature_Extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()

        )

    def forward(self,x):
        features = self.Feature_Extractor(x)
        #triplets = self.Triplet_Loss(features)
        return features



# def load_dataset(data_root, config):
#     # Placeholder for loading dataset - replace with actual implementation
#     transform = transforms.Compose([transforms.ToTensor()])
#     train_dataset = HPDataset(data_root, train=True, transform=transform)
#     test_dataset = HPDataset(data_root, train=False, transform=transform)
#     train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
#     return train_loader, test_loader

def train_nn(model, train_loader, device, num_epochs=30):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

def evaluate_nn(model, test_loader, device):
    model.eval()
    y_pred = []
    y_test = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            y_pred.extend(outputs.cpu().numpy())
            y_test.extend(batch_y.numpy())
    y_pred = np.array(y_pred) > 0.5
    y_test = np.array(y_test)
    return y_pred, y_test

def eval_with_data(X_train, y_train, X_test, y_test, device, config, net_name):

    if net_name!='own':
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
        input_size = X_train.shape[1]
        small_nn = SimpleNN(input_size).to(device)
    # Initialize and train the small neural network
    
    if net_name=='own': 
        train_dataset = pd.read_excel('../HPyloriData/HP_WSI-CoordAnnotatedWindows.xlsx')
        train_dataset = train_dataset[train_dataset['Deleted']==0][train_dataset['Cropped']==0][train_dataset['Presence']!=0].reset_index()

        gss = GroupShuffleSplit(n_splits=1, test_size=0.2,random_state=202)

        train_idx, test_idx = next(gss.split(X=train_dataset, y=train_dataset['Presence'], groups=train_dataset['Pat_ID']))
        train_loader = train_dataset.iloc[train_idx]
        valid_loader = train_dataset.iloc[test_idx]
        train_loader = train_loader.reset_index().drop('level_0',axis=1)
        valid_loader = valid_loader.reset_index().drop('level_0',axis=1)

        train_data_path = '../HPyloriData/annotated_windows'
        train_set = HPDataset(train_loader,path=train_data_path,train=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((32,32)),
                                                                                                        transforms.Normalize([0.8061, 0.8200, 0.8886], [0.0750, 0.0563, 0.0371])]))
        valid_set = HPDataset(valid_loader,path=train_data_path,train=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((32,32)),
                                                                                                        transforms.Normalize([0.8061, 0.8200, 0.8886], [0.0750, 0.0563, 0.0371])]))
        
        train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
        test_loader = DataLoader(valid_set, batch_size=config['batch_size'], shuffle=False)
        small_nn = ResNet_Triplet().to(device)
        
    train_nn(small_nn, train_loader, device)

    # Evaluate the small neural network
    y_pred, y_test = evaluate_nn(small_nn, test_loader, device)

    # Calculate and display confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig('confusion_matrix_nn_' + net_name + '.png')

    y_pred, y_test = evaluate_nn(small_nn, train_loader, device)

    # Calculate and display confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig('train_confusion_matrix_nn_' + net_name + '.png')

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)
    # folder_name = './runs/May17_20-55-07_userme'
    # checkpoints_folder = os.path.join(folder_name, 'checkpoints')
    # config = yaml.load(open(os.path.join(checkpoints_folder, "config.yaml"), "r"), Loader=yaml.FullLoader)
    # data_root = './data/'
    # model = load_model(checkpoints_folder, device)  # Assuming load_model is defined elsewhere in your code

    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    print(config)

    DataDir=r'C:\Users\xavis\OneDrive\Escritorio\Uni\TFG\ContrastiveHPyloriTFG\helico'
    data=np.load(os.path.join(DataDir,'PreTrainedFeatures'+'_Immuno_Aug'+'.npz'),
             allow_pickle=True)
    feResAug=data['feRes']
    feDensAug=data['feDens']
    feVGGAug=data['feVGG']
    feEffAug=data['feEff']
    y_trueAug=data['y_true']


    data=np.load(os.path.join(DataDir,'PreTrainedFeatures'+'_Immuno'+'.npz'),
                allow_pickle=True)
    y_true=data['y_true']
    pats = data['PatID_patches']
    feRes=data['feRes']
    
    cv = StratifiedGroupKFold(n_splits=6)

    train_idx, test_idx = next(cv.split(feRes,y_true,pats))

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2,random_state=202)
    train_idx, test_idx = next(gss.split(X=feRes, y=y_true, groups=pats))

    X_train = feRes[train_idx]
    X_test = feRes[test_idx]
    y_train = y_true[train_idx]
    y_test = y_true[test_idx]

    # gss = GroupShuffleSplit(n_splits=1, test_size=0.2,random_state=37)
    # train_idx, test_idx = next(gss.split(X=train_dataset, y=train_dataset['Presence'], groups=train_dataset['Pat_ID']))

    eval_with_data(X_train, y_train, X_test, y_test, device, config, net_name='ResNet')

    feDens=data['feDens']
    X_train = feDens[train_idx]
    X_test = feDens[test_idx]
    eval_with_data(X_train, y_train, X_test, y_test, device, config, net_name='DensNet')

    feVGG=data['feVGG']
    X_train = feVGG[train_idx]
    X_test = feVGG[test_idx]
    eval_with_data(X_train, y_train, X_test, y_test, device, config, net_name='VGG')


    feEff=data['feEff']
    X_train = feEff[train_idx]
    X_test = feEff[test_idx]
    eval_with_data(X_train, y_train, X_test, y_test, device, config, net_name='feEff')

    eval_with_data(X_train, y_train, X_test, y_test, device, config, net_name='own')


    # Example usage of eval_with_data
    # Assuming X_train, y_train, X_test, y_test are already defined
    # eval_with_data(X_train, y_train, X_test, y_test, device, config, net_name='ResNet')
