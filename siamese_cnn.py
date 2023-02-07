import keras
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras import models
from keras import layers
from tensorflow.keras import backend as K
import numpy as np

(train_X, train_Y), (test_X, test_Y) = fashion_mnist.load_data()

train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X / 255.
test_X = test_X / 255.

if K.image_data_format() == "channels_first":
    train_X = train_X.reshape((train_X.shape[0], 1, 28, 28))
    test_X = test_X.reshape((test_X.shape[0], 1, 28, 28))
else:
    train_X = train_X.reshape((train_X.shape[0], 28, 28, 1))
    test_X = test_X.reshape((test_X.shape[0], 28, 28, 1))

labels_train = train_Y
labels_test = test_Y
train_Y = to_categorical(train_Y, 10)
test_Y = to_categorical(test_Y, 10)

print(train_X.shape, test_X.shape)
print(train_Y.shape, test_Y.shape)

IMG_WIDTH = 28
IMG_HEIGHT = 28
IMG_DEPTH = 1
NUM_CLASSES = 10
BATCH_SIZE = 16
NUM_EPOCHS = 3
INIT_LR = 0.001


def VGG16():
    model = models.Sequential()
    inputShape = (IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH)
    chanDim = -1
    model.add(layers.Conv2D(32, (3, 3), padding="same",
                            input_shape=inputShape))
    model.add(layers.Activation("relu"))
    model.add(layers.BatchNormalization(axis=chanDim))
    model.add(layers.Conv2D(32, (3, 3), padding="same"))
    model.add(layers.Activation("relu"))
    model.add(layers.BatchNormalization(axis=chanDim))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64, (3, 3), padding="same"))
    model.add(layers.Activation("relu"))
    model.add(layers.BatchNormalization(axis=chanDim))
    model.add(layers.Conv2D(64, (3, 3), padding="same"))
    model.add(layers.Activation("relu"))
    model.add(layers.BatchNormalization(axis=chanDim))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())
    model.add(layers.Dense(512))
    model.add(layers.Activation("relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(NUM_CLASSES))
    model.add(layers.Activation("softmax"))

    model.summary()

    opt = keras.optimizers.SGD(lr=INIT_LR, momentum=0.9, decay=INIT_LR / NUM_EPOCHS)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

    history = model.fit(train_X, train_Y, batch_size=BATCH_SIZE, epochs=3)

    plt.plot(history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def siamese():
    def build_siamese_model(inputShape, embeddingDim=48):
        inputs = layers.Input(inputShape)

        x = layers.Conv2D(64, (2, 2), padding="same", activation="relu")(inputs)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Conv2D(64, (2, 2), padding="same", activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Dropout(0.3)(x)

        pooledOutput = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Dense(embeddingDim)(pooledOutput)
        model = keras.Model(inputs, outputs)
        return model

    def make_pairs(images, labels):
        pairImages = []
        pairLabels = []

        idx = [np.where(labels == i)[0] for i in range(0, NUM_CLASSES)]
        for idxA in range(len(images)):
            currentImage = images[idxA]
            label = labels[idxA]

            idxB = np.random.choice(idx[label])
            posImage = images[idxB]

            pairImages.append([currentImage, posImage])
            pairLabels.append([1]*10)

            negIdx = np.where(labels != label)[0]
            negImage = images[np.random.choice(negIdx)]

            pairImages.append([currentImage, negImage])
            pairLabels.append([0]*10)
        return np.array(pairImages), np.array(pairLabels)

    def distance(vectors):
        (featsA, featsB) = vectors

        sumSquared = K.sum(K.square(featsA - featsB), axis=1,
                           keepdims=True)
        return K.sqrt(K.maximum(sumSquared, K.epsilon()))

    pairTrain, labelTrain = make_pairs(train_X, labels_train)
    pairTest, labelTest = make_pairs(test_X, labels_test)

    imgA = layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH))
    imgB = layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH))
    featureExtractor = build_siamese_model((IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH))
    extr1 = featureExtractor(imgA)
    extr2 = featureExtractor(imgB)

    distance = layers.Lambda(distance)([extr1, extr2])
    outputs = layers.Dense(10, activation="sigmoid")(distance)
    model = keras.Model(inputs=[imgA, imgB], outputs=outputs)

    model.compile(loss="binary_crossentropy", optimizer="adam",
                  metrics=["accuracy"])

    print("[INFO] training model...")
    history = model.fit(
        [pairTrain[:, 0], pairTrain[:, 1]], labelTrain[:],
        validation_data=([pairTest[:, 0], pairTest[:, 1]], labelTest[:]),
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS)

    labels = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
              5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}

    x_test_features = model.predict([pairTest[:, 0], pairTest[:, 1]], verbose=True,
                                    batch_size=BATCH_SIZE)

    from sklearn.manifold import TSNE
    tsne_obj = TSNE(n_components=2,
                    init='pca',
                    random_state=101,
                    method='barnes_hut',
                    n_iter=250,
                    verbose=2)
    tsne_features = tsne_obj.fit_transform(x_test_features)
    obj_categories = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
                      'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
                      ]
    colors = plt.cm.rainbow(np.linspace(0, 1, 10))
    plt.figure(figsize=(10, 10))
    # y_test = labels_test.values.astype('int')
    for c_group, (c_color, c_label) in enumerate(zip(colors, obj_categories)):
        plt.scatter(tsne_features[np.where(labels_test == c_group), 0],
                    tsne_features[np.where(labels_test == c_group), 1],
                    marker='o',
                    color=c_color,
                    linewidth='1',
                    alpha=0.8,
                    label=c_label)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('t-SNE on Testing Samples')
    plt.legend(loc='best')
    plt.savefig('clothes-dist_trained.png')
    plt.show(block=False)


if __name__ == '__main__':
    VGG16()
