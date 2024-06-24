# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 17:19:56 2023

@author: debora
"""
import torch
from PreTrainedModels import *
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from ismember import ismember

# from skimage import measure

# from datasets import create_dataloader
# from test_models import eval_model

def create_dataloader(ims, labels, transform, batch_size, shuffle=False):
    """
    Creates a DataLoader for the given images and labels.

    Args:
    - ims (numpy.ndarray): Array of images.
    - labels (numpy.ndarray): Array of labels.
    - transform (callable, optional): Optional transform to be applied on a sample.
    - batch_size (int): How many samples per batch to load.
    - shuffle (bool, optional): Set to True to have the data reshuffled at every epoch.

    Returns:
    - DataLoader: DataLoader object for the dataset.
    """
    ims = torch.tensor(ims, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)

    if transform:
        ims = transform(ims)

    dataset = TensorDataset(ims, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader

def eval_model(model, dataloader):
    """
    Evaluates the model on the given DataLoader and extracts features.

    Args:
    - model (torch.nn.Module): The feature extraction model.
    - dataloader (DataLoader): DataLoader object for the dataset.

    Returns:
    - List: List of extracted features.
    """
    model.eval()
    features = []
    with torch.no_grad():
        for ims, _ in dataloader:
            ims = ims.to('cuda')
            outputs = model(ims)
            features.append(outputs.cpu().numpy())
    
    features = np.concatenate(features, axis=0)
    return features

def FeatExtractor(ims,Model,batch_size):
  

  # Instantation of Feature Extractor 
  
  if (Model == 'Vgg16' or Model == 'Vgg19'):   net = FeatureExtractorVGG(Model, cut_index=3)
  if (Model == 'DenseNet'):   net = FeatureExtractorDENSENET169(Model)
  if (Model == 'ResNet'):   net = FeatureExtractorResnet152(Model)
  if (Model == 'EfficientNet'):   net = FeatureExtractorEFFICIENTNETB7(Model)
  
  net.to('cuda')
  net.eval()
  
  fe=[]
  
  test_dataloader = create_dataloader(ims, np.ones(ims.shape[0]), 
                                        None, 
                                        batch_size,shuffle=False)
  
  fe=eval_model(net, test_dataloader)
  
  
  return fe

