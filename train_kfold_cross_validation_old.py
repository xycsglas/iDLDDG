

import tensorflow as tf
import numpy as np
import gc
from model import get_model
import os
import pandas as pd
import sys
from sklearn.model_selection import ShuffleSplit, StratifiedKFold, train_test_split
from sklearn.model_selection import StratifiedShuffleSplit, KFold
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression  # 引入线性回归模型
import openpyxl as op
import matplotlib.pyplot as plt
import warnings
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import statistics

warnings.filterwarnings("ignore")



SaveExcelName = "DNA276_10fold_221_2_2_2.xlsx"
TypeOfExperiment = "Esm1VEsmPtDNA276_good"



filename = f'D:\\MutsExperiment\\{TypeOfExperiment}\\{SaveExcelName}'


def op_toexcel(data, filename):
    if os.path.exists(filename):
        wb = op.load_workbook(filename)
        ws = wb.worksheets[0]
        ws.append(data)  # 每次写入一行
        wb.save(filename)

    else:
        wb = op.Workbook()
        ws = wb['Sheet']
        ws.append(['MSE', 'MAE', 'RMSE', 'R2', 'PCC', 'P_value'])
        ws.append(data)
        wb.save(filename)


def data_generator(train_esm, train_esm1v, train_prot, train_y, batch_size):
    L = train_esm.shape[0]

    while True:
        for i in range(0, L, batch_size):
            batch_esm = train_esm[i:i + batch_size].copy()
            batch_esm1v = train_esm1v[i:i + batch_size].copy()
            batch_prot = train_prot[i:i + batch_size].copy()
            batch_y = train_y[i:i + batch_size].copy()

            yield ([batch_esm, batch_esm1v, batch_prot], batch_y)


def cross_validation(train_esm, train_esm1v, train_prot, train_label, valid_esm, valid_esm1v, valid_prot, valid_label,
                     test_esm, test_esm1v, test_prot,
                     test_label, k, i):
    # 训练、验证each epoch的步长
    train_size = train_label.shape[0]
    val_size = valid_label.shape[0]
    batch_size = 16
    train_steps = train_size // batch_size
    val_steps = val_size // batch_size

    print(
        f"\n{k}Fold sample numbers：Training samples: {train_esm.shape[0]},Valid samples: {valid_esm.shape[0]} ,Test samples: {test_esm.shape[0]}")
    print("Validation samples:", len(valid_label))
    print(valid_label)
    print("Test samples:", len(test_label))
    print(test_label)


    qa_model = get_model()

    valiBestModel = f'D:\\MutsExperiment\\{TypeOfExperiment}\\model_10fold_181_SSDNA54_{i}_{k}.h5'


    checkpointer = tf.keras.callbacks.ModelCheckpoint(
        filepath=valiBestModel,
        monitor='val_root_mean_squared_error',
        save_weights_only=True,
        verbose=1,
        save_best_only=True,
        mode='min',
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_root_mean_squared_error',
        patience=80,
        verbose=0,
        mode='min'
    )

    train_generator = data_generator(train_esm, train_esm1v, train_prot,  train_label, batch_size)
    val_generator = data_generator(valid_esm, valid_esm1v, valid_prot, valid_label, batch_size)

    history_callback = qa_model.fit(
        train_generator,
        steps_per_epoch=train_steps,
        epochs=150,
        verbose=1,
        callbacks=[checkpointer, early_stopping],
        validation_data=val_generator,
        validation_steps=val_steps,
        shuffle=True,
        workers=1
    )

    rmse = history_callback.history['root_mean_squared_error']
    val_rmse = history_callback.history['val_root_mean_squared_error']
    epochs = range(1, len(rmse) + 1)

    plt.plot(epochs, rmse, 'bo', label='Training rmse')
    plt.plot(epochs, val_rmse, 'b', label='Validation rmse')
    plt.title(f'Test{i}Fold{k} Training And Validation rmse')
    plt.xlabel('150 Epochs 80 patience')
    plt.ylabel('RMSE')
    plt.legend()
    save_path = f'new_checkpoint1e-5221_bc16_120_Test{i}_Fold{k}_Training_And_Validation_rmse.png'
    plt.savefig(save_path)
    plt.clf()

    loss = history_callback.history['loss']
    val_loss = history_callback.history['val_loss']
    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')

    plt.title(f'TEST{i}Fold{k} Training And Validation LOSS')
    plt.xlabel('150 Epochs 80 patience')
    plt.ylabel('Loss')
    plt.legend()
    save_path = f'new_checkpoint1e-5221_bc16_120_TEST{i}Fold{k}_Training_And_Validation_LOSS.png'
    plt.savefig(save_path)
    plt.clf()


    train_generator.close()
    val_generator.close()
    del train_generator
    del val_generator
    gc.collect()

    print(
        f"\n{k}fold validation set result：Validation Loss: {history_callback.history['val_loss'][-1]:.4f}," + f"Validation RMSE: {history_callback.history['val_root_mean_squared_error'][-1]:.4f}")

    print(f"Fold {k} - Testing:")
    print(test_esm.shape)
    print(test_esm1v.shape)
    print(test_prot.shape)
    test_pred = qa_model.predict([test_esm, test_esm1v, test_prot]).reshape(-1, )

    evaluate_regression(test_pred, test_label)


    y_true_flat = np.ravel(test_label)
    y_pred_flat = np.ravel(test_pred)
    result_df = pd.DataFrame({
        'TRUE_label': y_true_flat,
        'PRED_label': y_pred_flat
    })
    result_end = fr'D:\\MutsExperiment\\{TypeOfExperiment}\\10k_181_label_{i}_{k}.csv'
    result_df.to_csv(result_end, index=False)


def evaluate_regression(test_pred, test_label):
    y_pred = test_pred
    y_true = test_label

    print("True value：" + str(test_label))
    print("Predicted value：" + str(test_pred))

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    pearsonr_corr, p_value = pearsonr(y_true, y_pred)

    print(f"\nFold {k} - Mean Squared Error (MSE): {mse:.4f}")
    print(f"Fold {k} - Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Fold {k} - RMSE: {rmse:.4f}")
    print(f"Fold {k} - R-squared (R2) Score: {r2:.4f}")
    print(f"Fold {k} - PCC: {pearsonr_corr:.4f}")
    print(f"Fold {k} - P-value: {p_value:.4f}")

    result = mse, mae, rmse, r2, pearsonr_corr, p_value
    op_toexcel(result, filename)


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    """    
    Esm2FileName = "ssDNA54_esm2_181.npy"
    Esm1vFileName = "ssDNA54_esm1v_181.npy"
    PTFileName = "ssDNA54_PT_181.npy"
    LabelFileName = "ssDNA54_181_label.npy"  
    """
    # Esm2FileName = "S552_181_esm.npy"
    # Esm1vFileName = "S552_181_esm1v.npy"
    # PTFileName = "S552_PT_181.npy"
    # LabelFileName = "S552_181_label.npy"

    Esm2FileName = "DNA552_esm2_181.npy"
    Esm1vFileName = "DNA552_esm1v_181.npy"
    PTFileName = "DNA552_PT_181.npy"
    LabelFileName = "DNA552_181_label.npy"

    all_esm = np.lib.format.open_memmap(f'./features_npy/{Esm2FileName}')
    all_esm_1v = np.lib.format.open_memmap(f'./features_npy/{Esm1vFileName}')
    all_prot = np.lib.format.open_memmap(f'./features_npy/{PTFileName}')
    all_label = np.lib.format.open_memmap(f'./features_npy/{LabelFileName}')


    all_label = all_label.astype(np.float)
    print(all_label.dtype)


    for i in range(1, 11):
        cv = KFold(n_splits = 10, shuffle=True, random_state = 42)
        k = 1
        for train_index, test_index in cv.split(all_esm, all_label):
            train_ESM = all_esm[train_index]
            train_ESM1V = all_esm_1v[train_index]
            train_Prot = all_prot[train_index]
            train_Y = all_label[train_index]

            train_ESM, valid_ESM, train_ESM1V, valid_ESM1V, train_Prot, valid_Prot, train_Y, valid_Y = train_test_split(
                train_ESM, train_ESM1V, train_Prot,
                train_Y, test_size=0.1,
                random_state = 42)

            test_ESM = all_esm[test_index]
            test_ESM1V = all_esm_1v[test_index]
            test_Prot = all_prot[test_index]
            test_Y = all_label[test_index]
            

            cross_validation(train_ESM, train_ESM1V, train_Prot, train_Y, valid_ESM, valid_ESM1V,
                             valid_Prot,  valid_Y,
                             test_ESM, test_ESM1V, test_Prot,
                             test_Y, k, i)

            k += 1
