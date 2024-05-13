import torch
import torch.nn as nn

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()
    
    
class ExponentialLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ExponentialLoss, self).__init__()
        self.margin = margin
    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        # losses = torch.relu(distance_positive - distance_negative + self.margin)
        losses = torch.exp(distance_positive-distance_negative + self.margin)
        return losses.mean()
    

class NTLogistic(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTLogistic,self).__init__()
        self.temp = temperature
    def calc_sigma(self, x):
        return None
    

class LNCE(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTLogistic,self).__init__()
        self.temp = temperature
    def calc_h(self, vI: torch.tensor, vIt: torch.tensor, vIp: torch.tensor) -> torch.tensor:

        term1 = torch.exp(torch.nn.functional.cosine_similarity(vI, vIt)/self.temp)
        term2 = torch.sum(torch.exp(torch.nn.functional.cosine_similarity(vIt.unsqueeze(0), vIp)))
        return (term1)/(term1-term2)

    def forward(self, vI: torch.tensor, vIt: torch.tensor, vIp: torch.tensor) -> torch.tensor:
        
        return (-torch.log(self.calc_h(vI,v)))

class NTXent(nn.Module):
    def __init__(self,temperature=5):
        super(NTXent, self).__init__()
        self.temp = temperature
    
    def product(self, x1 ,x2):
        suma=0
        for i in range(x1.shape[0]):
            suma+=torch.tensordot(x1[i], x2[i],1)
        return suma/self.temp
    
    def forward(self, anchor, positive, negative) -> torch.Tensor:

        pos = self.product(anchor,positive)

        neg = self.product(anchor,negative)
        return -(pos - torch.log(torch.exp(pos)+torch.exp(neg)))
    
# def nt_xent_loss(z_A, z_B, temperature=0.07):
#     batch_size = z_A.shape[0]
#     similarity = torch.exp(torch.mm(z_A, z_B.t().contiguous()) / temperature)
#     mask = torch.eye(batch_size, dtype=torch.float32).bool().cuda()
#     numerator = torch.sum(similarity * mask, dim=1)
#     denominator = torch.sum(similarity * ~mask, dim=1)
#     loss = -torch.log(numerator / denominator)
#     return loss.mean()
