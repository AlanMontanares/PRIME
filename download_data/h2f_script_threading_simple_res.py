import numpy as np
import pandas as pd
from tqdm import tqdm
import threading
import time
import os
import shutil

import astropy.units as u
from astropy.coordinates import Longitude, Latitude, Angle

from astroquery.hips2fits import hips2fits


def get_sdss_img(df, id, fov, size):
    
    ra,dec = np.float32(df.iloc[id][["ra","dec"]].values)

    x = hips2fits.query(
        hips="CDS/P/SDSS9/u",
        width=size,
        height=size,
        ra=Longitude(ra * u.deg),
        dec=Latitude(dec * u.deg),
        fov=Angle(fov * u.deg),
        projection="TAN",
        get_query_payload=False,
        format='fits',
        )


    g = hips2fits.query(
        hips="CDS/P/SDSS9/g",
        width=size,
        height=size,
        ra=Longitude(ra * u.deg),
        dec=Latitude(dec * u.deg),
        fov=Angle(fov  * u.deg),
        projection="TAN",
        get_query_payload=False,
        format='fits',
        )

    r = hips2fits.query(
        hips="CDS/P/SDSS9/r",
        width=size,
        height=size,
        ra=Longitude(ra * u.deg),
        dec=Latitude(dec* u.deg),
        fov=Angle(fov  * u.deg),
        projection="TAN",
        get_query_payload=False,
        format='fits',
        ) 

    i = hips2fits.query(
        hips="CDS/P/SDSS9/i",
        width=size,
        height=size,
        ra=Longitude(ra * u.deg),
        dec=Latitude(dec* u.deg),
        fov=Angle(fov * u.deg),
        projection="TAN",
        get_query_payload=False,
        format='fits',
        ) 

    z = hips2fits.query(
        hips="CDS/P/SDSS9/z",
        width=size,
        height=size,
        ra=Longitude(ra * u.deg),
        dec=Latitude(dec* u.deg),
        fov=Angle(fov * u.deg),
        projection="TAN",
        get_query_payload=False,
        format='fits',
        ) 

    x = x[0].data.byteswap().newbyteorder()
    g = g[0].data.byteswap().newbyteorder()
    r = r[0].data.byteswap().newbyteorder()
    i = i[0].data.byteswap().newbyteorder()
    z = z[0].data.byteswap().newbyteorder()

    x = np.nan_to_num(x, 0)
    g = np.nan_to_num(g, 0)
    r = np.nan_to_num(r, 0)
    i = np.nan_to_num(i, 0)
    z = np.nan_to_num(z, 0)

    img = np.zeros((5,size,size))

    img[0] = x
    img[1] = g
    img[2] = r
    img[3] = i
    img[4] = z

    return img


def download_batch(data_frame,inicio, final,name_dataset):

    stack = []
    max_retry = 2

    for x in tqdm(range(inicio,final)):

        for retry in range(max_retry):
            try:
                img = get_sdss_img(data_frame, x, fov=25.344/3600, size=64)
                stack.append(img)
                break

            except:
                if retry+1 == max_retry:
                    stack.append(np.zeros((5,64,64), dtype=np.float32))

    np.save(f'{name_dataset}/{name_dataset}_{final}.npy', np.stack(stack))


def download_all(df, name_dataset):

    os.makedirs(name_dataset, exist_ok=True)

    batch_size = 250
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
        stack = np.load(f'{name_dataset}/{name_dataset}_{batch}.npy')
        full.append(stack)
        
    full_final = np.concatenate(full, axis=0)
    np.save(f'h2f_sdss_{name_dataset}.npy', full_final.astype(np.float32))

    shutil.rmtree(name_dataset)

if __name__ == "__main__":
    
    df_dev = pd.read_csv("cat_train_9.csv")
    #df_test = pd.read_csv("cat_train_6.csv")
    #df_t1 = pd.read_csv("cat_train_7.csv")
    #df_t2 = pd.read_csv("cat_train_8.csv")

    alpha = time.time()

    t1 = threading.Thread(target=download_all, args=[df_dev, "train9"])
    #t2 = threading.Thread(target=download_all, args=[df_test, "train6"])
    #t3 = threading.Thread(target=download_all, args=[df_t1, "train7"])
    #t4 = threading.Thread(target=download_all, args=[df_t2, "train8"])

    t1.start()
    #t2.start()
    #t3.start()
    #t4.start()

    t1.join()
    #t2.join()
    #t3.join()
    #t4.join()

    print(f"Fin Total: {time.time()-alpha} [s]")