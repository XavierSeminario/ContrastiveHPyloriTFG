import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
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
            nn.Flatten()
            
        )

        self.vectorize = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )


    def forward(self, x):
        h = self.Feature_Extractor(x)
        # h = h.squeeze()

        x = self.vectorize(h)

        # x = self.l1(h)
        # x = F.relu(x)
        # # x = self.dropout(x)
        # # x = self.li(x)
        # # x = F.relu(x)
        # # x = self.dropout(x)
        # x = self.l2(x)
        # return x.squeeze(), x
        return h, x

if __name__ == "__main__": 
    model = ResNetSimCLR('resnet50',64)
