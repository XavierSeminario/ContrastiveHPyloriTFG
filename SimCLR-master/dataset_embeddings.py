import torch
from torch.utils.data import Dataset
import numpy as np
import random
import copy

class EmbeddingsDataset(Dataset):
    def __init__(self, embeddings, y_true, train=True):
        super(EmbeddingsDataset, self).__init__()
        self.embeddings = embeddings
        self.labels = y_true
        self.train = train
        
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        anchor_item = self.embeddings[idx]
        label = self.labels[idx]
        positive_list = self.embeddings[self.labels == label]
        negative_list = self.embeddings[self.labels != label]
        positive_item = random.choice(positive_list)
        negative_item = random.choice(negative_list)
        negative_item2 = random.choice(negative_list)

        # label = (label + 1)/2

        if self.train==True:
            if label == 1:
                aux = copy.deepcopy(positive_item)
                aux_2 = copy.deepcopy(anchor_item)
                anchor_item=copy.deepcopy(negative_item)
                negative_item=copy.deepcopy(aux_2)
                positive_item=copy.deepcopy(negative_item2)
                label=0
                negative_item2 = copy.deepcopy(aux)
            return (torch.tensor(anchor_item), torch.tensor(positive_item), torch.tensor(negative_item), torch.tensor(negative_item2)), torch.tensor(label)

        return (torch.tensor(anchor_item), torch.tensor(positive_item), torch.tensor(negative_item), torch.tensor(negative_item2)), torch.tensor(label), 3

# Example usage
# npz_file_path = '/mnt/data/embeddings.npz'  # Replace with your .npz file path
# dataset = EmbeddingsDataset(npz_file_path)

# # Example to get the first sample
# embedding, label = dataset[0]
# print(f'Embedding: {embedding}')
# print(f'Label: {label}')

# # Example to create a DataLoader
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# # Example to iterate through the DataLoader
# for batch_embeddings, batch_labels in dataloader:
#     print(f'Batch embeddings shape: {batch_embeddings.shape}')
#     print(f'Batch labels shape: {batch_labels.shape}')
#     break
