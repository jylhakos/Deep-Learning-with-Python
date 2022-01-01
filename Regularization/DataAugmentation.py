import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import numpy as np
import os
import matplotlib.pyplot as plt
import pathlib
import pandas as pd

# The path to the dataset
base_dir = pathlib.Path.cwd() / '..' / '..' /  '..' / 'coursedata' / 'cats_and_dogs_small'

# directories for training, validation and test splits
train_dir = base_dir / 'train'
validation_dir =  base_dir / 'validation'
test_dir = base_dir / 'test'

CLASS_NAMES = ['cats', 'dogs']
BATCH_SIZE = 32
IMG_SIZE = 150
EPOCHS = 20

def load_image(image_path):

    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image, channels=3) 
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = (image / 255)
    parts = tf.strings.split(image_path, os.path.sep)
    one_hot = parts[-2] == CLASS_NAMES
    label = tf.argmax(one_hot)

    return (image, label)

def configure_for_performance(ds, shuffle=False):
    
    if shuffle:
        ds = ds.shuffle(buffer_size=2000)

    ds = ds.batch(batch_size=BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    
    return ds

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = tf.data.Dataset.list_files(str(base_dir/'train/*/*.jpg'))
val_ds   = tf.data.Dataset.list_files(str(base_dir/'validation/*/*.jpg'))
test_ds  = tf.data.Dataset.list_files(str(base_dir/'test/*/*.jpg'))

with tf.device('/cpu:0'):

    train_ds = train_ds.map(load_image, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(load_image, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.map(load_image, num_parallel_calls=AUTOTUNE)

    train_ds = configure_for_performance(train_ds, shuffle=True)
    val_ds   = configure_for_performance(val_ds)
    test_ds  = configure_for_performance(test_ds)

#3 blocks of:
#Conv2D layer with 32 units, kernel size (3,3), activation ReLU
#Max pooling layer, kernel size (2,2)
#flattening layer
#dense layer with 128 units and ReLU activation
#output layer with 1 unit and sigmoid activation
#Use train_ds and val_ds for training (about 20 epochs)

model = keras.Sequential([
    layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", name="cv1"),
    layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", name="cv2"),
    layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", name="cv3"),
    layers.MaxPooling2D(pool_size=2, name="maxpool"),
    layers.Flatten(name="flatten"),
    layers.Dense(128, activation="relu", name="dense"),
    layers.Dense(1, activation='softmax', name="output")
])

model.summary()

tf.keras.utils.plot_model(
    model, 
    show_shapes=True, 
    show_layer_names=True
)

history = model.fit(X_train, y_train, batch_size=32,epochs=20,verbose=1,validation_data=(X_val, y_val))
model.save('model.h5')

# compile the model
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(),
              metrics='accuracy')


if training:
    history = model.fit(train_ds, batch_size=5,epochs=20,verbose=1,validation_data=val_ds)    model.save('model.h5')
else:
    model = tf.keras.models.load_model("model.h5")


