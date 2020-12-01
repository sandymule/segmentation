import tensorflow as tf
from tensorflow_addons.metrics import F1Score
import matplotlib.pyplot as plt
import time
import numpy as np


def load_image_train():
    input_path = "/data2/samwwong/FetalLung/preprocessed/train.npz"
    concat = np.load(input_path)

    concat_input_np = concat["inputs"]
    concat_segs_np = concat["segs"]

    return concat_input_np, concat_segs_np


def load_image_test():
    input_path = "/data2/samwwong/FetalLung/preprocessed/test.npz"
    concat = np.load(input_path)

    concat_input_np = concat["inputs"]
    concat_segs_np = concat["segs"]

    return concat_input_np, concat_segs_np


def create_datasets(X_train, y_train, X_test, y_test):
    X_train = X_train.reshape(X_train.shape[2], X_train.shape[0], X_train.shape[1], -1)
    y_train = y_train.reshape(y_train.shape[2], y_train.shape[0], y_train.shape[1], -1)

    X_test = X_test.reshape(X_test.shape[2], X_test.shape[0], X_test.shape[1], -1)
    y_test = y_test.reshape(y_test.shape[2], y_test.shape[0], y_test.shape[1], -1)

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(400).batch(64)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).shuffle(400).batch(64)

    return train_ds, test_ds


def unet2D():
    IMG_WIDTH = 256
    IMG_HEIGHT = 256
    IMG_CHANNELS = 1

    # Contracting Path
    inputs = tf.keras.layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
    c1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
                                activation='relu')(inputs)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
                                activation='relu')(c1)
    p1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
                                activation='relu')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
                                activation='relu')(c2)
    p2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(c2)

    c3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
                                activation='relu')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
                                activation='relu')(c3)
    p3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(c3)

    c4 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
                                activation='relu')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
                                activation='relu')(c4)
    p4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(c4)

    c5 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
                                activation='relu')(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
                                activation='relu')(c5)

    # Expansive Path
    u6 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
                                activation='relu')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
                                activation='relu')(c6)

    u7 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
                                activation='relu')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
                                activation='relu')(c7)

    u8 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
                                activation='relu')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
                                activation='relu')(c8)

    u9 = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=(2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
                                activation='relu')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
                                activation='relu')(c9)

    outputs = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(c9)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.summary()
    return model


def loss_fn(y_true, y_pred):
    loss = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=True)
    return loss


# @tf.function
def f1_score(y_true, y_pred):
    y_true = tf.cast(tf.keras.backend.flatten(y_true), tf.float32)
    y_pred = tf.cast(tf.keras.backend.flatten(y_pred), tf.float32)
    return 2 * (tf.keras.backend.sum(y_true * y_pred) + tf.keras.backend.epsilon()) / (
                tf.keras.backend.sum(y_true) + tf.keras.backend.sum(y_pred) + tf.keras.backend.epsilon())


# @tf.function
def train_step(images, labels, model, optimizer, train_loss):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_fn(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)

    return predictions


# @tf.function
def test_step(images, labels, model, test_loss):
    predictions = model(images, training=False)
    loss = loss_fn(labels, predictions)

    test_loss(loss)
    return predictions


# ############## CODE #################
X_train, y_train = load_image_train()
X_test, y_test = load_image_test()

train_ds, test_ds = create_datasets(X_train, y_train, X_test, y_test)

model = unet2D()
optimizer = tf.keras.optimizers.Adam()

EPOCHS = 20

train_loss = tf.keras.metrics.Mean(name='train_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')

for epoch in range(1, EPOCHS + 1):
    start_time = time.time()
    train_loss.reset_states()

    train_y_preds = []
    train_y_trues = []
    test_y_preds = []
    test_y_trues = []

    for train_images, train_labels in train_ds:
        print("TRAINING: EPOCH " + str(epoch))
        predictions = train_step(train_images, train_labels, model, optimizer, train_loss)
        train_y_trues.extend(train_labels)
        train_y_preds.extend(tf.keras.backend.round(predictions))

    train_y_trues = np.asarray(train_y_trues)
    train_y_preds = np.asarray(train_y_preds)

    train_f1_score = f1_score(train_y_trues, train_y_preds)
    print(train_y_preds)

    end_time = time.time()
    print("TRAINING TIME: ")
    print(end_time - start_time)

    print("TRAIN RESULTS: ")
    print(epoch, train_loss.result())
    print(epoch, train_f1_score)

    for test_images, test_labels in test_ds:
        predictions = test_step(test_images, test_labels, model, test_loss)
        test_y_trues.extend(test_labels)
        test_y_preds.extend(tf.keras.backend.round(predictions))

    test_y_trues = np.asarray(test_y_trues)
    test_y_preds = np.asarray(test_y_preds)

    test_f1_score = f1_score(test_y_trues, test_y_preds)

    print("TEST RESULTS: ")
    print(epoch, test_loss.result())
    print(epoch, test_f1_score)




