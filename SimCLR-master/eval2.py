import torch
import torch.nn as nn
import torch.optim as optim
import sys
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve
import pandas as pd
from sklearn.model_selection import train_test_split
from dataset_eval import HPDataset
from dataset_embeddings import EmbeddingsDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import GroupShuffleSplit
from sklearn.manifold import TSNE


# Define the small neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 1)  # Binary classification output
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# def load_dataset_old(data_root, config):
#     # Placeholder for loading dataset - replace with actual implementation
#     transform = transforms.Compose([transforms.ToTensor()])
#     train_dataset = HPDataset(data_root, train=True, transform=transform)
#     test_dataset = HPDataset(data_root, train=False, transform=transform)
#     train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
#     return train_loader, test_loader

def load_dataset(root ,config):
    train_dataset = pd.read_excel('../HPyloriData/HP_WSI-CoordAnnotatedWindows.xlsx')
    train_dataset = train_dataset[train_dataset['Deleted']==0][train_dataset['Cropped']==0][train_dataset['Presence']!=0].reset_index()
    # gss = GroupShuffleSplit(n_splits=1, test_size=0.2)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2,random_state=202)
    train_idx, test_idx = next(gss.split(X=train_dataset, y=train_dataset['Presence'], groups=train_dataset['Pat_ID']))

    # Crear los DataFrames de entrenamiento y prueba
    train_loader = train_dataset.iloc[train_idx]
    valid_loader = train_dataset.iloc[test_idx]
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
    
    data = np.load('C:/Users/xavis/OneDrive/Escritorio/Uni/TFG/ContrastiveHPyloriTFG/helico/PreTrainedFeatures_Immuno.npz')

    dades = data['feVGG']
    y_true = data['y_true']
    groups = data['PatID_patches']

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2,random_state=202)
    train_idx, test_idx = next(gss.split(X=dades, y=y_true, groups=groups))

    # Crear los DataFrames de entrenamiento y prueba

    train_set=EmbeddingsDataset(dades[train_idx],y_true[train_idx],train=False)
    valid_set=EmbeddingsDataset(dades[test_idx],y_true[test_idx],train=False)


    train_dl = DataLoader(train_set,batch_size=config['batch_size'],shuffle=True)
    test_dl = DataLoader(valid_set,batch_size=config['batch_size'],shuffle=True)
    return train_dl, test_dl

def train_nn(model, train_loader, device, num_epochs=30):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).float().unsqueeze(1)
            # print(batch_x.shape)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss/len(train_loader):.4f}')

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

def eval(model, data_root, device, config):
    train_loader, test_loader = load_dataset(data_root, config)

    X_train_feature = []
    y_train = []
    groups = []

    for (batch_x, _, batch_x_neg, _), batch_y, group in train_loader:
        batch_x = batch_x.to(device)
        print(batch_x.shape)
        features, _ = model(batch_x)
        # print(features.shape)
        X_train_feature.extend(features.cpu().detach().numpy())
        y_train.extend(batch_y.cpu().detach().numpy())
        groups.extend(group)
    
    X_test_feature = []
    y_test = []
    groups_test = []

    for (batch_x, _, batch_x_neg, _), batch_y, group in test_loader:
        batch_x = batch_x.to(device)
        features, _ = model(batch_x)
        X_test_feature.extend(features.cpu().detach().numpy())
        y_test.extend(batch_y.cpu().detach().numpy())
        groups_test.extend(group)

    X_train_feature = np.array(X_train_feature)
    X_test_feature = np.array(X_test_feature)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # full_patients = np.concatenate((groups,groups_test))
    full_diag = np.concatenate((X_train_feature ,X_test_feature))
    full_real = np.concatenate((y_train, y_test))

    tsne = TSNE(n_components=2, random_state=42, perplexity=20)
    embeddings_2d = tsne.fit_transform(full_diag)

    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=full_real, cmap='viridis', marker='o')
    plt.title('Visualització t-SNE dels Embedings 128-Dimensionals')
    plt.colorbar()
    plt.show()

    scaler = StandardScaler()
    scaler.fit(X_train_feature)
    X_train = scaler.transform(X_train_feature)
    X_test = scaler.transform(X_test_feature)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # Initialize and train the small neural network
    input_size = X_train.shape[1]
    small_nn = SimpleNN(input_size).to(device)
    train_nn(small_nn, train_loader, device)

    # Evaluate the small neural network
    y_pred, y_test = evaluate_nn(small_nn, test_loader, device)

    # Calculate and display confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig('confusion_matrix_nn.png')

    y_pred, y_test = evaluate_nn(small_nn, train_loader, device)

    # Calculate and display confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig('train_confusion_matrix_nn.png')


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)
    folder_name = './runs/May17_20-55-07_userme'
    checkpoints_folder = os.path.join(folder_name, 'checkpoints')
    config = yaml.load(open(os.path.join(checkpoints_folder, "config.yaml"), "r"), Loader=yaml.FullLoader)
    data_root = './data/'
    model = load_model(checkpoints_folder, device)  # Assuming load_model is defined elsewhere in your code
    eval(model, data_root, device, config)
