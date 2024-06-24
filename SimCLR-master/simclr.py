import torch
from models.resnet_simclr_old import ResNetSimCLR
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from loss.nt_xent import NTXentLoss
from loss.nt_logistic import NTLogisticLoss
from loss.marginal_tripletC import MarginalTripletLossC
import os
import shutil
import sys
from dataset import HPDataset
from dataset_embeddings import EmbeddingsDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit, train_test_split


apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp

    apex_support = True
except:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False

import numpy as np

torch.manual_seed(0)


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))


class SimCLR(object):

    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()
        self.writer = SummaryWriter()
        self.dataset = dataset
        self.model = ResNetSimCLR(**self.config["model"]).to(self.device)
        self.criterion = self._make_loss()
        self.batch_size = self.config['batch_size']
    def _make_loss(self):
        if (self.config['loss_type']=='nt_logistic'):
            return NTLogisticLoss(self.device,self.config['batch_size'], **self.config['loss'])
        elif (self.config['loss_type']=='nt_xent'):
            return NTXentLoss(self.device,self.config['batch_size'], **self.config['loss'])
        elif (self.config['loss_type']=='marginal_triplet'):
            return MarginalTripletLossC(self.device,self.config['batch_size'], **self.config['loss'])

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    def _step(self, model, xis, xjs, nis, njs, n_iter, labels):

        # get the representations and the projections
        ris, zis = model(xis)  # [N,C]
        # print(ris.shape)
        # get the representations and the projections
        rjs, zjs = model(xjs)  # [N,C]

        kis, yis = model(nis)  # [N,C]

        # get the representations and the projections
        kjs, yjs = model(njs)  # [N,C]
        

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        yis = F.normalize(yis, dim=1)
        yjs = F.normalize(yjs, dim=1)

        loss = self.criterion(zis, zjs, yis, yjs, labels)
        return loss

    def train(self):

        

        model = self.model
        model = self._load_pre_trained_weights(model)

        # trainable_parameters = model.get_trainable_parameters()
        # filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam( model.parameters(), 0.001, weight_decay=eval(self.config['weight_decay']))

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1500, eta_min=0,
                                                               last_epoch=-1)

        if apex_support and self.config['fp16_precision']:
            model, optimizer = amp.initialize(model, optimizer,
                                              opt_level='O2',
                                              keep_batchnorm_fp32=True)

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints_test')

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        train_data_path = '../HPyloriData/annotated_windows'

        tr, val, train_dataset = self.dataset.get_data_loaders()
        # print(epoch_counter)
        # print(splits)
        train_loader = tr.reset_index().drop('level_0',axis=1)
        # valid_loader = val.reset_index().drop('level_0',axis=1)
        train_loader, val_loader, _, _ = train_test_split(train_loader, train_loader['Presence'], test_size=0.1)

        train_loader = train_loader.reset_index().drop('level_0',axis=1)
        valid_loader = val_loader.reset_index().drop('level_0',axis=1)
        train_set = HPDataset(train_loader,path=train_data_path,train=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((32,32)),
                                                                                                        transforms.Normalize([0.8061, 0.8200, 0.8886], [0.0750, 0.0563, 0.0371])]))
        valid_set = HPDataset(valid_loader,path=train_data_path,train=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((32,32)),
                                                                                                        transforms.Normalize([0.8061, 0.8200, 0.8886], [0.0750, 0.0563, 0.0371])]))
        
        data = np.load('C:/Users/xavis/OneDrive/Escritorio/Uni/TFG/ContrastiveHPyloriTFG/helico/PreTrainedFeatures_Immuno_Aug.npz')

        dades = data['feRes']
        y_true = data['y_true']
        groups = data['PatID_patches']

        cv = StratifiedGroupKFold(n_splits=15)
        cont=1
        for train_idx, test_idx in cv.split(dades, y_true, groups):
            train_set=dades[train_idx]
            valid_set=dades[test_idx]
            y_data = y_true[train_idx]
            if cont==15:
                break
            cont+=1
           
        train_loader, valid_loader,y_train,y_valid = train_test_split(train_set, y_data, test_size=0.1)

        train_set = EmbeddingsDataset(train_loader,y_train)
        valid_set=EmbeddingsDataset(valid_loader,y_valid)
        # Crear los DataFrames de entrenamiento y prueba

        

        train_loader = DataLoader(train_set,batch_size=self.batch_size,shuffle=True,drop_last=True)
        valid_loader = DataLoader(valid_set,batch_size=self.batch_size,shuffle=False,drop_last=True)

        for epoch_counter in range(self.config['epochs']):
            print(epoch_counter)
            # splits, X, train_dataset = self.dataset.get_data_loaders()
            # print(epoch_counter)
            # # print(splits)
            # for train_index, test_index in splits:
            #     X_train, X_test = X[train_index], X[test_index]
            #     # print(train_dataset)
            #     train_loader = train_dataset[np.isin(X, X_train)].reset_index().drop('level_0',axis=1)
            #     valid_loader = train_dataset[np.isin(X, X_test)].reset_index().drop('level_0',axis=1)
            #     # color_jitter = transforms.ColorJitter(0 * self.s, 0.2 * self.s, 0.2 * self.s, 0 * self.s)
            #     train_set = HPDataset(train_loader,path=train_data_path,train=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((32,32)),
            #                                                                                                     transforms.Normalize([0.8061, 0.8200, 0.8886], [0.0750, 0.0563, 0.0371])]))
            #     valid_set = HPDataset(valid_loader,path=train_data_path,train=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((32,32)),
            #                                                                                                     transforms.Normalize([0.8061, 0.8200, 0.8886], [0.0750, 0.0563, 0.0371])]))

            #     train_loader = DataLoader(train_set,batch_size=self.batch_size,shuffle=True,drop_last=True)
            #     valid_loader = DataLoader(valid_set,batch_size=self.batch_size,shuffle=True,drop_last=True)
            for (xis, xjs, nis, njs), labels in train_loader:
                # print(xis.shape)
                optimizer.zero_grad()

                xis = xis.to(self.device)
                xjs = xjs.to(self.device)
                nis = nis.to(self.device)
                njs = njs.to(self.device)
                labels = labels.to(self.device)
                loss = self._step(model, xis, xjs, nis, njs, n_iter,labels)
                print(loss)
                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)

                if apex_support and self.config['fp16_precision']:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                n_iter += 1

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss = self._validate(model, valid_loader)
                if valid_loss < best_valid_loss:
                    # save the model weights
                    print(valid_loss)
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
                    # print('hola')
                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                scheduler.step()
            self.writer.add_scalar('cosine_lr_decay', scheduler.get_lr()[0], global_step=n_iter)

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./runs', self.config['fine_tune_from'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader):

        # validation steps
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            counter = 0
            print('Eval Loss:')
            for (xis, xjs,nis,njs), labels in valid_loader:
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                nis = nis.to(self.device)
                njs = njs.to(self.device)
                labels = labels.to(self.device)

                loss = self._step(model, xis, xjs, nis, njs, counter,labels)
                print(loss)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        model.train()
        return valid_loss
