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
        losses = torch.exp(distance_positive-distance_negative)
        return losses.mean()
    

import torch
import torch.nn.functional as F

def simclr_loss(anchor, positive, negative, temperature=0.5):
    """
    Calcula la pérdida SimCLR.
    
    Args:
    - anchor: Un tensor de forma (N, D), donde N es el tamaño del lote y D es la dimensión de las características.
    - positive: Un tensor de forma (N, D), donde N es el tamaño del lote y D es la dimensión de las características.
    - negative: Un tensor de forma (N, D), donde N es el tamaño del lote y D es la dimensión de las características.
    - temperature: Un escalar que escala los logits antes de que se aplique la operación softmax.
    
    Returns:
    - La pérdida SimCLR.
    """
    # Concatena las características de ancla, positivas y negativas
    features = torch.cat([anchor, positive, negative], dim=0)
    
    # Calcula la matriz de similitud
    sim_matrix = torch.mm(features, features.t())
    
    # Calcula las máscaras positivas y negativas
    pos_mask = torch.roll(torch.eye(3 * N), N, 1).bool().to(features.device)
    neg_mask = (~torch.eye(3 * N).bool()).to(features.device)
    
    # Calcula los logits
    pos_logits = torch.masked_select(sim_matrix, pos_mask).view(N, 3)
    neg_logits = torch.masked_select(sim_matrix, neg_mask).view(N, 3 * N - 3)
    
    # Calcula las pérdidas positivas y negativas
    pos_loss = F.cross_entropy(pos_logits / temperature, torch.zeros(N).long().to(features.device))
    neg_loss = F.cross_entropy(neg_logits / temperature, torch.zeros(N).long().to(features.device))
    
    # Devuelve la pérdida total
    return pos_loss + neg_loss

    
