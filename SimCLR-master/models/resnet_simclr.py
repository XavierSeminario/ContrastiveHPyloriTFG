import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        # super(ResNetSimCLR, self).__init__()
        # self.resnet_dict = {"resnet18": models.resnet18(pretrained=True),
        #                     "resnet50": models.resnet50(pretrained=True)}

        # resnet = self._get_basemodel(base_model)
        # if (base_model == "resnet50"):
        #     self.features = []
        #     for name, module in resnet.named_children():
        #         if name == 'conv1':
        #             module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        #         if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
        #             self.features.append(module)
        #     #print(self.features)
        #     self.features = nn.Sequential(*self.features)
            
        # num_ftrs = resnet.fc.in_features

        # for param in self.parameters():
        #     param.requires_grad = False

        # self.features = nn.Sequential(*list(resnet.children())[:-1])
        # #print(num_ftrs)
        # # projection MLP
        # self.l1 = nn.Linear(num_ftrs, 512)
        # self.li = nn.Linear(512,256)
        # self.l2 = nn.Linear(512, out_dim)
        # self.dropout = nn.Dropout(0.2) 

        # for param in self.l1.parameters():
        #     param.requires_grad = True
            
        # for param in self.li.parameters():
        #     param.requires_grad = True
        

        # for param in self.l2.parameters():
        #     param.requires_grad = True
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

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

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
    
class ResNet_Triplet(nn.Module):
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
            nn.Linear(2304, 512),
            nn.ReLU(),
            nn.Linear(512, 64)
        )
    def forward(self,x):
        features = self.Feature_Extractor(x)
        #triplets = self.Triplet_Loss(features)
        return features

if __name__ == "__main__": 
    model = ResNetSimCLR('resnet50',64)
