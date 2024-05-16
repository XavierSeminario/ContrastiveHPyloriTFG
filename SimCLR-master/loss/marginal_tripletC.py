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
        # print(zis.shape)
        # print(zjs.shape)
        # print(labels.shape)
        # representations = torch.cat([zjs, zis], dim=0)
        # repres = torch.cat([njs, nis], dim=0)
        # repres_neg = torch.cat([zis, nis], dim=0)
        # # representations = torch.cat([zis, zjs, nis], dim=0)
        # # repres = torch.cat([nis, njs, zis], dim=0)
        # similarity_matrix = self.similarity_function(representations, representations)
        # similarity = self.similarity_function(repres, repres)
        # similarity_neg = self.similarity_function(repres_neg, repres_neg)
        # # print(similarity_neg.shape)
        # # print(similarity_matrix.shape)
        # # filter out the scores from the positive samples
        # l_pos = torch.diag(similarity_matrix, self.batch_size)
        # r_pos = torch.diag(similarity_matrix, -self.batch_size)
        # l_neg = torch.diag(similarity, self.batch_size)
        # r_neg = torch.diag(similarity, -self.batch_size)
        # print('POS')
        # print(l_pos.shape)
        # print(r_pos.shape)
        # positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        positives_1 = self.similarity_function(zis,zjs)
        # print(positives_1.shape)
        positives_2 = self.similarity_function(nis,njs)
        negatives_1 = self.similarity_function(zis,nis)
        negatives_2 = self.similarity_function(nis,zis)

        positives = torch.cat([positives_1, positives_2]).view(2*self.batch_size, -1) 
        negatives = torch.cat([negatives_1, negatives_2]).view(2*self.batch_size, -1)
        # print(positives.shape)
        # print(positives.shape)
        # negatives = torch.cat([l_neg, r_neg]).view(2 * self.batch_size, 1) 
        # pr = similarity_neg[:self.batch_size,self.batch_size:]
        # br = similarity_neg[self.batch_size:,:self.batch_size]
        # negatives = torch.cat([pr, pr, br, br]).view(4 * self.batch_size, -1) 
        logits =negatives - positives + self.m

        # print(logits)
        # print(logits.shape)
        #print(logits.shape)
        loss = torch.sum(self.activation(logits))
        #print(loss)
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