import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import matplotlib.pyplot as plt # plotting library
from keras.src.datasets import mnist
from keras.src.saving import load_model
from tensorflow.python.keras.utils.np_utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_test = to_categorical(y_test)
image_size = x_train.shape[1]
input_size = image_size * image_size
x_test = np.reshape(x_test, [-1, input_size])
x_test = x_test.astype('float32') / 255


def eval(model):
    results = model.evaluate(x_test, y_test, batch_size=128)
    print("test loss, test acc:", results)


model = load_model('models/mnist.hdf5')
model.summary()

eval(model)


def compress(model):
    for l in model.layers:
        l = None
    return model


model = compress(model)