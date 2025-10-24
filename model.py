import tensorflow as tf
from Encoder import Encoder
from tensorflow import keras
import numpy as np
from sklearn.metrics import mean_squared_error
from Mutil_scale_prot import MultiScaleConvA
from tensorflow.keras.regularizers import l2


# 定义Pseudo-Huber Loss损失函数
def pseudo_huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    delta_squared = delta ** 2
    huber_loss = delta_squared * (tf.sqrt(1 + (error / delta) ** 2) - 1)
    return huber_loss


def get_model():
    inputESM = tf.keras.layers.Input(shape=(181, 1280))
    inputESM1V = tf.keras.layers.Input(shape=(181, 1280))
    inputProt = tf.keras.layers.Input(shape=(181, 1024))

    sequence = tf.keras.layers.Dense(512)(inputESM)
    sequence = tf.keras.layers.Dense(256)(sequence)
    sequence = Encoder(3, 256, 4, 1024, rate=0.3)(sequence)
    sequence = sequence[:,90, :]

    sequence_esm1v = tf.keras.layers.Dense(512)(inputESM1V)
    sequence_esm1v = tf.keras.layers.Dense(256)(sequence_esm1v)
    sequence_esm1v = Encoder(3, 256, 4, 1024, rate=0.3)(sequence_esm1v)
    sequence_esm1v = sequence_esm1v[:, 90, :]

    sequence_prot = tf.keras.layers.Dense(512)(inputProt)
    sequence_prot = tf.keras.layers.Dense(256)(sequence_prot)
    sequence_prot = Encoder(3, 256, 4, 1024, rate=0.3)(sequence_prot)
    Prot = sequence_prot[:, 90, :]
    #Prot = MultiScaleConvA()(sequence_prot)

    sequenceconcat = tf.keras.layers.Concatenate()([sequence, sequence_esm1v, Prot])

    l2_reg = 1e-3  # 设置为1e-4到1e-2之间

    # feature = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=l2(l2_reg))(sequenceconcat)
    # feature = tf.keras.layers.Dropout(0.4)(feature)
    feature = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=l2(l2_reg))(sequenceconcat)
    feature = tf.keras.layers.Dropout(0.3)(feature)
    feature = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=l2(l2_reg))(feature)
    feature = tf.keras.layers.Dropout(0.3)(feature)

    y = tf.keras.layers.Dense(1)(feature)
    qa_model = tf.keras.models.Model(inputs=[inputESM, inputESM1V, inputProt],
                                     outputs=y)  # 创建回归模型，并定义了输入层和输出层
    adam = tf.keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=1.0,
                                    clipvalue=0.5)  # 模型使用‘Adam’优化器进行编译

    delta = 1
    qa_model.compile(optimizer=adam, loss=lambda y_true, y_pred: pseudo_huber_loss(y_true, y_pred, delta=delta),
                     metrics=['mse', 'mae', tf.keras.metrics.RootMeanSquaredError()])

    qa_model.summary()  # 打印出模型的摘要信息，包括每一层的名称，输出形状、参数数量等
    return qa_model  # 该函数返回了编译好的模型qa_model，在构建完模型并编译后，并将其返回以供训练和使用
