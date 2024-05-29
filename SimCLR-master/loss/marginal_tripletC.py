import torch
import numpy as np

class MarginalTripletLossC(torch.nn.Module):
    def __init__(self,device,batch_size,m,use_cosine_similarity):
        super(MarginalTripletLossC, self).__init__()
        self.batch_size = batch_size
        self.m = m
        self.device = device
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.activation = torch.nn.ReLU()
    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)
    
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
    def forward(self,zis,zjs,nis,njs,labels=None):
        positives_1 = self.similarity_function(zis,zjs)
        positives_2 = self.similarity_function(nis,njs)
        negatives_1 = self.similarity_function(zis,nis)
        negatives_2 = self.similarity_function(nis,zis)

        positives = torch.cat([positives_1, positives_2]).view(2*self.batch_size, -1) 
        negatives = torch.cat([negatives_1, negatives_2]).view(2*self.batch_size, -1)

        logits =negatives - positives + self.m

        loss = torch.sum(self.activation(logits))

        return loss / (self.batch_size*(self.batch_size-1))
if __name__ == "__main__":
    Loss = MarginalTripletLossC('cpu',4,1,True)
    print(Loss.mask_samples_from_same_repr)
    xi = torch.rand((4,10))
    xj = torch.rand((4,10))
    ni = torch.rand((4,10))
    nj = torch.rand((4,10))
    loss = Loss(xi,xj,ni,nj)
    print(loss)