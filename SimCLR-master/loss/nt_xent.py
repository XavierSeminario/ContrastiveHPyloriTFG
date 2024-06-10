import torch
import numpy as np
import torch.nn.functional as F


class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity


    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def _normalize_embeddings(self, *embeddings):
        return [F.normalize(embed, p=2, dim=1) for embed in embeddings]

    def forward(self, zis, zjs, nis, njs, label=None, device=None):
        zis, zjs, nis, njs = self._normalize_embeddings(zis, zjs, nis, njs)

        
        positives_1 = torch.diag(self.similarity_function(zis,zjs))
        positives_2 = torch.diag(self.similarity_function(nis,njs))
        negatives_1 = self.similarity_function(zis,nis)
        negatives_2 = self.similarity_function(nis,zis)

        positives = torch.cat([positives_1, positives_2]).view(2*self.batch_size, 1) 
        negatives = torch.cat([negatives_1, negatives_2]).view(2*self.batch_size, -1)
        # negatives, _ = torch.max(negatives, dim=1, keepdim=True)
        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2*self.batch_size).to(self.device).long()

        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)
if __name__ == "__main__":
    Loss = NTXentLoss('cpu',4,0.5,True)
    print(Loss.mask_samples_from_same_repr)
    xi = torch.rand((4,10))
    xj = torch.rand((4,10))
    loss = Loss(xi,xj)
    print(loss)