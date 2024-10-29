import lightning as L
import os
import torch
import torch.nn as nn
import torch.optim as optim


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, nbS1, nbS2, with_kernel_5=True):
        super(InceptionBlock, self).__init__()

        self.conv1_1 = nn.Conv2d(in_channels, nbS1, kernel_size=1, stride=1, padding="same")
        self.conv1_3 = nn.Conv2d(in_channels, nbS1, kernel_size=1, stride=1, padding="same")

        self.conv2_1 = nn.Conv2d(in_channels, nbS2, kernel_size=1, stride=1, padding="same")
        self.conv2_2 = nn.Conv2d(nbS1, nbS2, kernel_size=3, stride=1, padding="same")

        if with_kernel_5:
            self.conv1_2 = nn.Conv2d(in_channels, nbS1, kernel_size=1, stride=1, padding="same")
            self.conv2_3 = nn.Conv2d(nbS1, nbS2, kernel_size=5, stride=1, padding="same")

        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)#AvgPool2dSamePadding(kernel_size=2, stride=1)
        
        self.prelu1 = nn.PReLU(num_parameters=nbS1, init=0.0)
        self.prelu2 = nn.PReLU(num_parameters=nbS2, init=0.0)

        self.with_kernel_5 = with_kernel_5

    def forward(self,x):
        s1_1 = self.prelu1(self.conv1_1(x))
        s1_3 = self.prelu1(self.conv1_3(x))

        s2_1 = self.prelu2(self.conv2_1(x))
        s2_2 = self.prelu2(self.conv2_2(s1_1))
        
        s2_4 = self.avgpool(s1_3)
        
        if self.with_kernel_5:
            s1_2 = self.prelu1(self.conv1_2(x))
            s2_3 = self.prelu2(self.conv2_3(s1_2))
            
            return torch.cat((s2_1, s2_2, s2_3, s2_4), dim=1)
        
        else:
            return torch.cat((s2_1, s2_2, s2_4), dim=1)


class PRIME(L.LightningModule):

    def __init__(self, config):
        super(PRIME, self).__init__()

        self.convs = nn.Sequential(
                                nn.Conv2d(config["in_channels"], 64, kernel_size=5, padding="same"),
                                nn.PReLU(num_parameters=64, init=0.0),
                                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

                                InceptionBlock(64,nbS1=48, nbS2=64, with_kernel_5=True),
                                InceptionBlock(240,nbS1=64, nbS2=92, with_kernel_5=True),
                                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

                                InceptionBlock(340,nbS1=92, nbS2=128, with_kernel_5=True),
                                InceptionBlock(476,nbS1=92, nbS2=128, with_kernel_5=True),
                                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

                                InceptionBlock(476,nbS1=92, nbS2=128, with_kernel_5=False),
                                nn.Flatten()
                                )
        
        self.linear_net = nn.Sequential(
                                nn.Linear(int(((config["img_size"]/8)**2)*348)*config["in_levels"] +1, 1096),
                                nn.ReLU(),
                                nn.Linear(1096, 1096),
                                nn.ReLU(),
                                nn.Linear(1096, 180)
                                )
                                
        #self.initialize_weights()
        
        if config["loss"] == "ce":
            self.loss = nn.CrossEntropyLoss()

        self.lr = config["lr"]
        self.in_levels= config["in_levels"]

        self.training_predictions = []
        self.training_classes = []

        self.val_predictions = []
        self.val_classes = []

        self.curves = {
                        "train_loss": [],
                        "val_loss": [],
                    }
        
        self.config = config
        self.save_hyperparameters(config)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight) #Glorot Uniform
                nn.init.constant_(m.bias, 0.1)
                
    def forward(self, x, ebv):

        x = torch.cat([self.convs(x[:,i,:,:,:]) for i in range(self.in_levels)] + [ebv], dim=1)

        output = self.linear_net(x)
        return output
    
    def training_step(self, batch, batch_idx):
        x,ebv,y_class = batch
        x_hat = self.forward(x,ebv)
        train_loss = self.loss(x_hat, y_class)

        self.training_predictions.append(x_hat.detach().cpu())
        self.training_classes.append(y_class.cpu())
        
        self.log("train_loss", train_loss, prog_bar=True)
        return train_loss

    def validation_step(self, batch, batch_idx):

        x,ebv,y_class = batch
        x_hat = self.forward(x,ebv)
        val_loss = self.loss(x_hat, y_class)

        self.val_predictions.append(x_hat.cpu())
        self.val_classes.append(y_class.cpu())

        self.log("val_loss", val_loss, prog_bar=True)
        
        return val_loss
    
    def on_train_epoch_end(self):

        predictions = torch.cat(self.training_predictions, dim=0)
        classes = torch.cat(self.training_classes, dim=0)

        self.curves["train_loss"].append(self.loss(predictions, classes))

        self.training_predictions.clear() 
        self.training_classes.clear()  

    def on_validation_epoch_end(self):

        predictions = torch.cat(self.val_predictions, dim=0)
        classes = torch.cat(self.val_classes, dim=0)

        self.curves["val_loss"].append(self.loss(predictions, classes))

        self.val_predictions.clear() 
        self.val_classes.clear() 

    def predict_step(self, batch):
        x,ebv,_ = batch    
        return self(x,ebv)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=0.0, last_epoch=-1)

        return [optimizer], [scheduler]