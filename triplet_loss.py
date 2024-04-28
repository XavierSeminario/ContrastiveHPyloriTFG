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

    
