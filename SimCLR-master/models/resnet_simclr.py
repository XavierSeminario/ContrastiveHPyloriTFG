import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super().__init__()
        self.Feature_Extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )

    def forward(self, x):
        h = self.Feature_Extractor(x)
        x = h.squeeze()

        # x = self.l1(h)
        # x = F.relu(x)
        # # x = self.dropout(x)
        # # x = self.li(x)
        # # x = F.relu(x)
        # # x = self.dropout(x)
        # x = self.l2(x)
        # return x.squeeze(), x
        return h, h

if __name__ == "__main__": 
    model = ResNetSimCLR('resnet50',64)
