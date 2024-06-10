import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.l0 = nn.Linear(4096,4096)
        self.l1 = nn.Linear(4096, 512)
        self.l2 = nn.Linear(512, out_dim)

    def forward(self, x):
        h = self.l0(x)
        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)
        return h,x
if __name__ == "__main__":
    model = ResNetSimCLR('resnet50',64)