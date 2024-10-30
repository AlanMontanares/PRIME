import numpy as np
import pandas as pd
from tqdm import tqdm
import threading
import time
#import xarray as xr

from panstamps.downloader import downloader
import logging
import os
import shutil

from astropy.io import fits

def get_panstamps_img(df, id, fov, size, name_dataset):
    
    ra,dec = df.iloc[id][["ra","dec"]].values
    objid = df["objID"][id]

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    name_stack, _, _ = downloader(
        log=logger,
        settings=False,
        downloadDirectory=os.path.join(name_dataset, str(objid)),
        fits=True,
        jpeg=False,
        arcsecSize=fov,
        filterSet='grizy',
        color=False,
        singleFilters=True,
        ra=str(ra),
        dec=str(dec),
        imageType="stack",  # warp | stack
        mjdStart=False,
        mjdEnd=False,
        window=False
                ).get()

    img = np.zeros((5,size,size), dtype=np.float32)

    if len(name_stack) == 5:
        for i in range(5):
            data = fits.open(name_stack[i])
            data = data[0].data.byteswap().newbyteorder()
            data = np.nan_to_num(data, 0)
            img[i] = data

    shutil.rmtree(os.path.join(name_dataset, str(objid)))

    return img

def download_batch(data_frame,inicio, final,name_dataset):

    stack = []
    max_retry = 2
    for x in tqdm(range(inicio,final)): 

        for retry in range(max_retry):
            try:
                img = get_panstamps_img(data_frame, x, fov=24, size=96, name_dataset=name_dataset)
                stack.append(img)
                break  

            except:
                if retry+1 == max_retry:
                    stack.append(np.zeros((5, 96, 96), dtype=np.float32))

    np.save(f'{name_dataset}/stack_{final}.npy', np.stack(stack))


def download_all(df, name_dataset):

    batch_size = 1000
    start = 0
    total = len(df)

    arr = np.arange(start, total, batch_size)
    arr = np.append(arr, total)

    n_procesos = len(arr)-1
    threads = []

    ini = time.time()

    for i in range(n_procesos):
        stop = min(arr[i]+batch_size, total)
        t = threading.Thread(target=download_batch, args=[df, arr[i], stop, name_dataset])
        t.start()
        threads.append(t)

    for thread in threads:
        thread.join()

    print(f"Dataset Finalizado en {time.time()-ini} [s]")

    full = []
    for batch in arr[1:n_procesos+1]:
        stack = np.load(f'{name_dataset}/stack_{batch}.npy')
        full.append(stack)
        
    full_final = np.concatenate(full, axis=0)
    np.save(f'panstamps_{name_dataset}.npy', full_final.astype(np.float32))

    shutil.rmtree(name_dataset)


if __name__ == "__main__":
    

    df_train1 = pd.read_csv("cat_train_8.csv")
    df_train2 = pd.read_csv("cat_train_9.csv")

    alpha = time.time()

    t1 = threading.Thread(target=download_all, args=[df_train1, "train_dataset_8"])
    t2 = threading.Thread(target=download_all, args=[df_train2, "train_dataset_9"])

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    print(f"Fin Total: {time.time()-alpha} [s]")