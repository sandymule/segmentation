import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import time

# from absl import flags
# from absl import app
# from absl import logging

# flags.DEFINE_integer('batch_size', 1, 'Batch size')
# flags.DEFINE_integer('num_epochs', 1, 'Number of epochs')
# flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate for training')
# flags.DEFINE_string('ckpt_path', 'checkpoints/unet3d_', 'Checkpoint Directory')
# flags.DEFINE_integer('seed', 0, 'Seed for shuffling training batch')
#
# flags.DEFINE_string('noisy_path', '/home/data2/liztong/AI_rsFMRI/Resampled/Resampled_noise', 'Directory with Noisy fMRI Images')
# flags.DEFINE_string('clean_path', '/home/data2/liztong/AI_rsFMRI/Resampled/Resampled_clean_v2', 'Directory with Clean fMRI Images')
# flags.DEFINE_integer('train_len', 400, 'Number of 3D-images per training example')
#

dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask

@tf.function
def load_image_train(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

def load_image_test(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 64
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE


train = dataset['train'].map(load_image_train, num_parallel_calls = tf.data.experimental.AUTOTUNE)
test = dataset['test'].map(load_image_test)

train_dataset = train.cache().shuffle(BATCH_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test.batch(BATCH_SIZE)



######################## DEFINE THE MODEL ###########################

def unet():
    IMG_WIDTH = 128
    IMG_HEIGHT = 128
    IMG_CHANNELS = 3

    #Contracting Path
    inputs = tf.keras.layers.Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
    s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)
    c1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(s)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(c1)
    p1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(c2)
    p2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(c2)

    c3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(c3)
    p3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(c3)

    c4 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(c4)
    p4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(c4)

    c5 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(c5)

    #Expansive Path
    u6 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(filters = 128, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(filters = 128, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(c6)

    u7 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(filters = 64, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(filters = 64, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(c7)

    u8 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(filters = 32, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(filters = 32, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(c8)

    u9 = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=(2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis = 3)
    c9 = tf.keras.layers.Conv2D(filters = 16, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(filters = 16, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(c9)

    outputs = tf.keras.layers.Conv2D(filters = 1, kernel_size=(1, 1), activation = 'sigmoid')(c9)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.summary()
    return model


def loss_fn(y_true, y_pred):
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
    return loss(y_true, y_pred)


# @tf.function
def train_step(images, labels, model, optimizer, train_loss):
    with tf.GradientTape() as tape:
        predictions = model(images, training = True)
        loss = loss_fn(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)


# @tf.function
def test_step(images, labels, model, test_loss):
    with tf.GradientTape() as tape:
        predictions = model(images, training=False)
        loss = loss_fn(labels, predictions)

    test_loss(loss)


############## CODE #################
model = unet()
optimizer = tf.keras.optimizers.Adam()

EPOCHS = 20
OUTPUT_CHANNELS = 3

train_loss = tf.keras.metrics.Mean(name = 'train_loss')
#train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name = 'train_accuracy')
test_loss = tf.keras.metrics.Mean(name = 'test_loss')
#test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name = 'test_accuracy')

for epoch in range(1, EPOCHS + 1):
    start_time = time.time()
    train_loss.reset_states()
    # train_accuracy.reset_states()
    test_loss.reset_states()
    # test_accuracy.reset_states()

    for train_images, train_labels in train_dataset:
        train_step(train_images, train_labels, model, optimizer, train_loss)

    end_time = time.time()
    print(end_time - start_time)

    print(epoch, train_loss.result())
    # for test_images, test_labels in test_dataset:
    #     test_step(test_images, test_labels, model, test_loss)






