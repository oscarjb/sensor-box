import numpy as np
import tensorflow as tf
# from tensorflow import keras
import os

model = tf.keras.models.load_model(os.path.join('server/static/model/model.h5'), compile=False)


def pad(A, length=10000):
    arr = np.zeros(length)
    arr[:len(A)] = A
    return arr


def predict(sample):
    sample_pad = np.array(pad(sample))
    return int(round(model.predict(np.array([sample_pad]))[0][0]))
    print("test")