# Photometric Redshift Inference via Multi-resolution imaging Evaluation (PRIME)
## Carpeta `pasquet_vs_hips2fits`
En esta carpeta se encuentran los resultados obtenidos al entrenar el modelo propuesto por Pasquet et al. 2019, pero usando dos fuentes de datos. La primera son los mismos datos utilizados por Pasquet, y la otra fuente utilizada fue HiPS2FITS. El entrenamiento fue identico para ambos, mismos datasets, mismos hiperparametros, etc.

En esta carpeta se encuentran los siguientes archivos:
- **`datasets.py`**: Este archivo Python procesa y crea los dataloaders para entrenar el modelo. Los dataloaders quedan dentro de un `LightningDataModule`, el cual contendrá los datasets de entrenamiento, validación y test. El dataset de entrenamiento posee Data-augmentation, el cual consiste en rotaciones aleatorias en 0°, 90°, 180° y 270° grados además de Flips horizontales y verticales. Las imágenes no fueron pre-procesadas.
- **`model_pasquet.py`**: Este archivo Python contiene el modelo propuesto por Pasquet et al. 2019 en el framework de PyTorch Lightning. La gracia de este framework es que tiene todas las fases necesarias para entrenar en métodos de la misma clase donde creamos el modelo mediante el módulo de Lightning (`LightningModule`). Adicionalmente se agregan 3 métodos:
  - `on_train_epoch_end()`: En este método, al final de cada época, se guarda la loss de entrenamiento promedio de la época en un diccionario.
  - `on_validation_epoch_end()`: Cumple el mismo labor que el metodo anterior, pero adicionalmente se van guardando los zphot en validación en cada final de época.
  - `predict_step()`: Este método es utilizado para predecir los zphot mediante el dataset de test ubicado en el `LightningDataModule`. Ojo, muy importante.
- **`utils.py`**: En este script se tienen funciones para visualizar los resultados, como curvas de entrenamiento, scatter plot de la regresion, etc.
- **`train_pasquet.py`**: Finalmente, en este archivo se utilizan todos los scripts anteriores para efectuar el entrenamiento de forma automatica. Para tener mayor control de este se utiliza la libreria **argparse**, mediante la cual se podran modificar todos los hiperparametros del entrenamiento. Adicionalmente, todos los resultados son almacenados en una carpeta ubicada en una direccion entregada por el usuario (en caso de no existir la carpeta se creará automaticamente).

Para poder llevar a cabo el entrenamiento se da un ejemplo de como se debe ejecutar el script `train_pasquet.py`:

```python
python train_pasquet.py --train_path datasets\sdss_train.npz --val_path datasets\sdss_val.npz --test_path datasets\sdss_test.npz --epoch 30 --save_files resultados/sdss_128 --seed 48 --num_workers 11 --batch_size 128


**NOTA**: El archivo `plot_comparacion_sdss_vs_h2f.ipynb` contiene los graficos de comparaciones obtenidas al usar ambos tipos de datos (Pasquet vs HiPS2FITS).
