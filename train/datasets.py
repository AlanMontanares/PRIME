import torch
import random

import numpy as np
import lightning as L
import torchvision.transforms.functional as TF

from typing import Sequence
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset



class CustomDataset(Dataset):
    def __init__(self,imgs,ebv,z_class, transform=None):
        self.imgs = imgs
        self.ebv =ebv
        self.z_class = z_class
        self.transform = transform

    def __len__(self):
        return len(self.z_class)

    def __getitem__(self, idx):

        image = self.imgs[idx]
        
        if self.transform:
            image = self.transform(image)

        return image, self.ebv[idx], self.z_class[idx]
    

class MyRotateTransform:
    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

class RandomResolution:
    def __init__(self, levels: Sequence[int]):
        self.levels = levels

    def __call__(self, x):
        indice = random.choice(self.levels)
        return x[indice-1, :, :, :]


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class GalaxyDataModule(L.LightningDataModule):
    
    def __init__(self, imgs_train,imgs_test,z_train_class,z_test_class,ebv_train,ebv_test, batch_size=128, seed=0, num_workers =4):
        super().__init__()
        self.batch_size = batch_size
        self.imgs_train = imgs_train
        self.imgs_test = imgs_test
        self.z_train_class = z_train_class
        self.z_test_class = z_test_class
        self.ebv_train = ebv_train
        self.ebv_test = ebv_test
        self.seed = seed
        self.num_workers = num_workers
        self.aug_transform = transforms.Compose([
                                        #RandomResolution(levels=[1,2,3,4,5]),
                                        MyRotateTransform(angles=[0,90,180,270]),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        #transforms.Resize((64,64))
                                        ])
    
                                            
    def setup(self, stage=None):
        
        self.train_dataset = CustomDataset(
          self.imgs_train,
          self.ebv_train,
          self.z_train_class,
          transform=self.aug_transform,
        )

        self.test_dataset = CustomDataset(
          self.imgs_test,
          self.ebv_test,
          self.z_test_class,
          #transform= transforms.Resize((64,64))
          )

    def train_dataloader(self):
        return DataLoader(
          self.train_dataset,
          batch_size=self.batch_size,
          shuffle=True,
          num_workers=self.num_workers,
          persistent_workers=True,
          pin_memory=False,
          worker_init_fn=seed_worker,
          generator = torch.Generator().manual_seed(self.seed)
        )

    def val_dataloader(self):
        return DataLoader(
          self.test_dataset,
          batch_size=self.batch_size,
          shuffle=False,
          num_workers=self.num_workers,
          persistent_workers=True,
          pin_memory=False,
          generator = torch.Generator().manual_seed(self.seed)
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=False,
            generator = torch.Generator().manual_seed(self.seed)
        )
    

def normalization(imgs_train,imgs_test,type=""):

    if type == "asinh":
        print("Asinh")
        imgs_train = torch.asinh(imgs_train)
        imgs_test = torch.asinh(imgs_test)

    elif type == "sqrt":
        print("SQRT")
        imgs_train = torch.sign(imgs_train)*(torch.sqrt(torch.sign(imgs_train)*(imgs_train/1000) + 1) - 1)
        imgs_test = torch.sign(imgs_test)*(torch.sqrt(torch.sign(imgs_test)*(imgs_test/1000) + 1) - 1)

    elif type =="linear":
        print("Lineal")
        imgs_train = 0.00015*imgs_train +0.01
        imgs_test = 0.00015*imgs_test +0.01  
        
    else:
        print("Sin pre-procesamiento")
        imgs_train = imgs_train
        imgs_test = imgs_test
            
    return imgs_train, imgs_test