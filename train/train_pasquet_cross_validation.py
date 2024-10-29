from datasets import *
from model_pasquet import *

import argparse
import os
import time

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default="datasets\h2f_multiresolution_train_n1_kfold1.npz", help='Train dataset path')
    parser.add_argument('--test_path', type=str, default="datasets\h2f_multiresolution_test_n1_kfold1.npz", help='Test dataset path')
    parser.add_argument('--normalization', type=str, default=None, help='Type of normalization on imgs')
    parser.add_argument('--loss', type=str, default="ce", help='Loss Function Train')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate Train')
    parser.add_argument('--in_channels', type=int, default=5, help='N° channels imgs')
    parser.add_argument('--img_size', type=int, default=64, help='Image size')
    parser.add_argument('--batch_size', type=int, default=128, help='Training Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Num of Workers of dataloaders')
    parser.add_argument('--epoch', type=int, default=40, help='Training Epochs')
    parser.add_argument('--save_files', type=str, default="resultados", help='File name of the results')
    parser.add_argument('--seed', type=int, default=0, help='Seed of the experiment')
 
    args = parser.parse_args()


#-----------REPRODUCIBILIDAD-----------#
    L.seed_everything(args.seed, workers=True)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("medium")
#-----------REPRODUCIBILIDAD-----------#

#-----------CARPETA CONTENEDORA-----------#
    os.makedirs(args.save_files, exist_ok=True)
#-----------CARPETA CONTENEDORA-----------#

#-----------CARGA DE DATOS-----------#
    print("Cargando Datos\n")

    inicio = time.time()

    range_z = np.linspace(0, 0.4, 181)[:-1]
    
    train = np.load(args.train_path, allow_pickle=True)
    test = np.load(args.test_path, allow_pickle=True)

    imgs_train = torch.tensor(train["imgs"]).float()
    z_train_class = torch.tensor(np.digitize(train["metadata"].item()["z"].values,range_z)-1).long()
    ebv_train = torch.tensor(train["metadata"].item()["EBV"].values).unsqueeze(1).float()

    del train

    imgs_test = torch.tensor(test["imgs"]).float()
    z_test_class = torch.tensor(np.digitize(test["metadata"].item()["z"].values,range_z)-1).long()
    ebv_test = torch.tensor(test["metadata"].item()["EBV"].values).unsqueeze(1).float()

    imgs_train, imgs_test = normalization(imgs_train,imgs_test,type=args.normalization)

    del test

    print(f"Carga de datos finalizada en {time.time()-inicio} [s]\n")
#-----------CARGA DE DATOS-----------#



#-----------ENTRENAMIENTO-----------#
    dm = GalaxyDataModule(imgs_train=imgs_train,
                        imgs_test=imgs_test,
                        z_train_class=z_train_class,
                        z_test_class=z_test_class,
                        ebv_train=ebv_train,
                        ebv_test=ebv_test,
                        batch_size=args.batch_size,
                        seed=args.seed,
                        num_workers=args.num_workers
                            )
    config ={
        "in_channels": args.in_channels,
        "loss": args.loss,
        "lr": args.lr,
        "img_size":args.img_size,
        "save_files":args.save_files}

    model_aug = Pasquet(config)

    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        monitor="val_loss", 
        dirpath=args.save_files, 
        #filename='pasquet_best_{epoch}', 
        #save_top_k=1,
        save_last = True,  
        mode="min")

    trainer = L.Trainer(
        num_sanity_val_steps=0,
        logger=False,
        deterministic=True,
        max_epochs=args.epoch,
        accelerator ="gpu",
        devices = "auto",
        callbacks=[checkpoint_callback])

    inicio = time.time()
    trainer.fit(model_aug, dm)
    print(f"Entrenamiento finalizado en {time.time()-inicio} [s]\n")
#-----------ENTRENAMIENTO-----------#

#-----------PREDICCIONES-----------#
    prob = nn.Softmax(dim=1)(torch.cat(trainer.predict(model=model_aug, datamodule =dm, ckpt_path="last"), dim=0))
    mid_point_z = (torch.linspace(0, 0.4, 181)[:-1] + torch.linspace(0, 0.4, 181)[1:]) / 2

    zphot  = (prob*mid_point_z).sum(1)
    np.save(os.path.join(args.save_files, "zphot_test_midpoint.npy"), zphot.numpy())
#-----------PREDICCIONES-----------#

#-----------RESULTADOS-----------#
    np.save(os.path.join(args.save_files, "curvas.npy"), model_aug.curves)
#-----------RESULTADOS-----------#
