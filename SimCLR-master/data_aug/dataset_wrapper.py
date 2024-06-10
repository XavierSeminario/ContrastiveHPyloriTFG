import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
# from data_aug.gaussian_blur import GaussianBlur
#from gaussian_blur import GaussianBlur
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.model_selection import train_test_split
from dataset import HPDataset


np.random.seed(0)


class DataSetWrapper(object):

    def __init__(self, batch_size, num_workers, valid_size, input_shape, s, use_aug):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.s = s
        self.use_aug = use_aug
        self.input_shape = eval(input_shape)

    def get_data_loaders(self):
        # data_augment = self._get_simclr_pipeline_transform()

        #train_dataset = datasets.STL10('/data2/meng/SimCLR/SimCLRpytorch/SimCLR/data/', split='train+unlabeled', download=False,
        #                               transform=SimCLRDataTransform(data_augment))
        # train_dataset = datasets.CIFAR10('./data/',download=True,transform=SimCLRDataTransform(data_augment))
        train_dataset = pd.read_excel('../HPyloriData/HP_WSI-CoordAnnotatedWindows.xlsx')
        train_dataset_aug = pd.read_excel('../HPyloriData/HP_WSI-CoordAugAnnotatedWindows.xlsx')

        train_dataset = train_dataset[train_dataset['Deleted']==0][train_dataset['Cropped']==0][train_dataset['Presence']!=0]
        if self.use_aug:
            train_dataset = train_dataset.drop(columns=['Deleted', 'Cropped'])
            train_dataset_aug = train_dataset_aug.drop(columns=['Unnamed: 0.1','Unnamed: 0'])

            train_dataset = pd.concat([train_dataset, train_dataset_aug], ignore_index=True)

            def add_leading_zeros(value):

                try:
                    num_part, text_part = value.split('_')
                    num_part = num_part.zfill(5)  # Pad the numeric part with leading zeros up to 5 digits
                    return f"{num_part}_{text_part}"
                except:
                    return str(value).zfill(5) 
            train_dataset['Window_ID'] = train_dataset['Window_ID'].apply(add_leading_zeros)
            
        train_dataset = train_dataset.reset_index()
        print(train_dataset)

        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader, train_dataset

    # def _get_simclr_pipeline_transform(self):
    #     # get a set of data augmentation transformations as described in the SimCLR paper.
    #     color_jitter = transforms.ColorJitter(0.4 * self.s, 0.4 * self.s, 0.4 * self.s, 0.1 * self.s)
    #     data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=self.input_shape[0]),
    #                                           transforms.RandomHorizontalFlip(),
    #                                           transforms.RandomApply([color_jitter], p=0.8),
    #                                           transforms.RandomGrayscale(p=0.2),
    #                                           #GaussianBlur(kernel_size=int(0.1 * self.input_shape[0])),
    #                                           transforms.ToTensor(),
    #                                           transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    #     return data_transforms

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        # gss = GroupShuffleSplit(n_splits=6, test_size=0.18)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2,random_state=202)
        train_idx, test_idx = next(gss.split(X=train_dataset, y=train_dataset['Presence'], groups=train_dataset['Pat_ID']))

        # Crear los DataFrames de entrenamiento y prueba
        train_loader = train_dataset.iloc[train_idx]
        valid_loader = train_dataset.iloc[test_idx]

        # print(train_loader.sort_values(by=['Pat_ID']))
        # print(valid_loader.sort_values(by=['Pat_ID']))
        # train_loader,valid_loader = train_test_split(train_dataset,test_size=0.2,stratify=train_dataset['Presence'])
        

        return train_loader, valid_loader
        # for train_index, test_index in splits:
        #     X_train, X_test = X[train_index], X[test_index]
        #     print(train_dataset)
        #     train_loader = train_dataset[np.isin(X, X_train)].reset_index().drop('level_0',axis=1)
        #     valid_loader = train_dataset[np.isin(X, X_test)].reset_index().drop('level_0',axis=1)

        # train_data_path = '../HPyloriData/annotated_windows'
        # color_jitter = transforms.ColorJitter(0 * self.s, 0.2 * self.s, 0.2 * self.s, 0 * self.s)
        # train_set = HPDataset(train_loader,path=train_data_path,train=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((32,32)),
        #                                                                                                 transforms.Normalize([0.8061, 0.8200, 0.8886], [0.0750, 0.0563, 0.0371])]))
        # valid_set = HPDataset(valid_loader,path=train_data_path,train=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((32,32)),
        #                                                                                                 transforms.Normalize([0.8061, 0.8200, 0.8886], [0.0750, 0.0563, 0.0371])]))

        # train_dl = DataLoader(train_set,batch_size=self.batch_size,shuffle=True,drop_last=True)
        # test_dl = DataLoader(valid_set,batch_size=self.batch_size,shuffle=True,drop_last=True)
        # return train_dl, test_dl


class SimCLRDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        xi = self.transform(sample)
        xj = self.transform(sample)
        return xi, xj
if __name__ == "__main__":
    dataset = DataSetWrapper(batch_size=1,num_workers=1,valid_size=0.05,input_shape='(32,32,3)',s=1)
    train_loader, valid_loader = dataset.get_data_loaders()
    print(train_loader)
