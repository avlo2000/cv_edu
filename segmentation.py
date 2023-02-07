from keras import layers
from keras import Model
from keras import optimizers

import matplotlib.pyplot as plt
import numpy as np
import tqdm

import cv2
import glob

SIZE = 50


def load_data():
    def split(img):
        return img[:, 0:256], img[:, 256:512]

    test_x = []
    test_y = []
    for filename in tqdm.tqdm(glob.glob('data/cityscapes_data/test/*.jpg')[:SIZE]):
        x, y = split(cv2.imread(filename))
        test_x.append(x)
        test_y.append(y)

    train_x = []
    train_y = []
    for filename in tqdm.tqdm(glob.glob('data/cityscapes_data/train/*.jpg')[:SIZE]):
        x, y = split(cv2.imread(filename))
        train_x.append(x)
        train_y.append(y)

    return np.stack(train_x, axis=0), np.stack(train_y, axis=0), np.stack(test_x, axis=0), np.stack(test_y, axis=0)


def build_model():
    input = layers.Input(shape=(256, 256, 3))

    x = layers.Conv2DTranspose(64, (3, 3), activation="relu", padding="same")(input)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Conv2DTranspose(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)

    x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same")(x)
    return Model(input, x)


model = build_model()
train_x, train_y, test_x, test_y = load_data()
model.compile(optimizer=optimizers.RMSprop(learning_rate=0.1, rho=0.8), loss="mse")

model.fit(
    x=train_x,
    y=train_y,
    epochs=10,
    batch_size=128,
    shuffle=True,
)
model.save("models/segmentator.hdf5")

preds = model.predict(test_x)

for p in preds:
    plt.imshow(p)
    plt.show()

