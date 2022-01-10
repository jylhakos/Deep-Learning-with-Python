import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import pathlib
import pandas as pd
import numpy as np
import os

# Import the pre-trained VGG16 model from Keras
from tensorflow.keras.applications import VGG16

# A function to preprocess input for futher feeding into VGG16 network
from tensorflow.keras.applications.vgg16 import preprocess_input

from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation, RandomZoom

from tensorflow.keras.layers import AveragePooling2D

# Set training=True, when training network
training=True

CLASS_NAMES = ['cats', 'dogs']
BATCH_SIZE = 32
IMG_SIZE = 60

AUTOTUNE = tf.data.experimental.AUTOTUNE

def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image, channels=3) 
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = preprocess_input(image)
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

# Import model from tensorflow.keras.applications
from tensorflow.keras.applications import VGG16

from tensorflow import keras
from tensorflow.keras import layers

# Pass arguments
conv_base = VGG16(weights=None, include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

head = keras.Sequential()

head.add(layers.Flatten(input_shape=conv_base.output_shape[1:], name="flatten"))

head.add(layers.Dense(4096, activation="relu", name="dense"))

model = tf.keras.models.Model(conv_base.input, head(conv_base.output))

model.summary()

model = tf.keras.models.Sequential()

model.add(conv_base)

flatten_layer = layers.Flatten(input_shape=(BATCH_SIZE, conv_base.output_shape[1:]), name="flatten")

model.add(flatten_layer)

model.add(layers.Dense(128, activation="relu", name="dense"))

model.add(layers.Dense(1, activation='sigmoid', name="output"))

model.summary()

# Pre-trained VGG16 model as a feature extractor

base_dir = pathlib.Path.cwd() / '..' / '..' /  '..' / 'coursedata' / 'cats_and_dogs_small'

# directories for training,
# validation and test sets
train_dir = base_dir / 'train' 
validation_dir =  base_dir / 'validation'
test_dir = base_dir / 'test'


AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = tf.data.Dataset.list_files(str(base_dir/'train/*/*.jpg'), shuffle=False)
val_ds   = tf.data.Dataset.list_files(str(base_dir/'validation/*/*.jpg'), shuffle=False)
test_ds  = tf.data.Dataset.list_files(str(base_dir/'test/*/*.jpg'), shuffle=False)


with tf.device('/cpu:0'):
    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    train_ds = train_ds.map(load_image, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(load_image, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.map(load_image, num_parallel_calls=AUTOTUNE)

    train_ds = train_ds.batch(batch_size=1997)
    val_ds   = val_ds.batch(batch_size=995)
    test_ds  = test_ds.batch(batch_size=1000)

def get_labels(ds):
    for images, labels in ds.take(1):
        return labels

# get training labels
train_labels = get_labels(train_ds)

# get validation labels
val_labels = get_labels(val_ds)

# get test labels
test_labels = get_labels(test_ds)

# check number of labels
print("Number of training labels: ", len(train_labels))
print("Number of validation labels: ", len(val_labels))
print("Number of test labels: ", len(test_labels))

train_ds = tf.data.Dataset.list_files(str(base_dir/'train/*/*.jpg'), shuffle=False)
val_ds   = tf.data.Dataset.list_files(str(base_dir/'validation/*/*.jpg'), shuffle=False)
test_ds  = tf.data.Dataset.list_files(str(base_dir/'test/*/*.jpg'), shuffle=False)


with tf.device('/cpu:0'):
    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    train_ds = train_ds.map(load_image, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(load_image, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.map(load_image, num_parallel_calls=AUTOTUNE)

    train_ds = configure_for_performance(train_ds)
    val_ds   = configure_for_performance(val_ds)
    test_ds  = configure_for_performance(test_ds)

# Load pre-trained VGG16 conv base from keras
conv_base = VGG16(weights ='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

conv_base.trainable = False

#for layer in conv_base.layers[:19]:
#    layer.trainable = False

for i, layer in enumerate(conv_base.layers):
    print(i, layer.name, layer.trainable)

conv_base.summary()


train_features = conv_base.predict(train_ds)
val_features   = conv_base.predict(val_ds)
test_features  = conv_base.predict(test_ds)

print(train_features.shape)
print(val_features.shape)
print(test_features.shape)

# Train classification head on extracted features.

clf_head = tf.keras.models.Sequential([
                                      layers.Flatten(input_shape=(1,1,512), name="flatten"),
                                      layers.Dense(128, activation="relu", name="dense"),
                                      layers.Dense(1, activation='sigmoid', name="output")
                                      ])

clf_head.summary()

# Compile the model 
clf_head.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.RMSprop(),
                  metrics='accuracy')

# Training the model
if training:
    history = clf_head.fit(train_features, train_labels, batch_size = 32, epochs=20, validation_data=(val_features, val_labels))
    clf_head.save('clf_head.h5')
else: 
    clf_head = tf.keras.models.load_model("clf_head.h5")

fig, ax = plt.subplots(1,2, figsize=(10,3))
df_accuracy = pd.DataFrame(history.history).loc[:,['accuracy','val_accuracy']]
df_loss = pd.DataFrame(history.history).loc[:,['loss','val_loss']]

df_accuracy.plot(ax=ax[0])
df_loss.plot(ax=ax[1])
ax[0].set_ylim(0.5,1.05)
ax[1].set_ylim(-0.5,5)

plt.show()

test_loss, test_acc = clf_head.evaluate(test_features, test_labels)
print(f'The test set accuracy of model is {test_acc:.2f}')

# Fine-tuning pre-trained model.

train_ds = tf.data.Dataset.list_files(str(base_dir/'train/*/*.jpg'))
val_ds   = tf.data.Dataset.list_files(str(base_dir/'validation/*/*.jpg'))
test_ds  = tf.data.Dataset.list_files(str(base_dir/'test/*/*.jpg'))

with tf.device('/cpu:0'):
    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    train_ds = train_ds.map(load_image, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(load_image, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.map(load_image, num_parallel_calls=AUTOTUNE)

    train_ds = configure_for_performance(train_ds, shuffle=True)
    val_ds   = configure_for_performance(val_ds)
    test_ds  = configure_for_performance(test_ds)


data_augmentation = tf.keras.Sequential(
    [
        RandomFlip("horizontal", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        RandomRotation(0.1, fill_mode='constant'),
        RandomZoom(0.1,0.1, fill_mode='constant')
    ]
)

# Create Sequential object
model = tf.keras.models.Sequential()

# Add data_augmentation block
data_augmentation.summary()
model.add(data_augmentation)

# Add conv base
conv_base.summary()

model.add(conv_base)

# Add clf_head
clf_head = tf.keras.models.Sequential([
                                      layers.Flatten(input_shape=(1, 1, 512), name="flatten"),
                                      layers.Dense(128, activation="relu", name="dense"),
                                      layers.Dense(1, activation='sigmoid', name="output")
                                      ])

clf_head.summary()
model.add(clf_head)

model.summary()

# Compile the model 
model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.RMSprop(),
                  metrics='accuracy')

# model training ~10 min 
if training:
    #history = model.fit(train_features, train_labels, batch_size = 32, epochs=2, validation_data=(val_features, val_labels))
    history = model.fit(train_ds, batch_size = 32, epochs=20, validation_data=val_ds)
    model.save('model.h5')

# unfreeeze all layers
conv_base.trainable = True

for i, layer in enumerate(conv_base.layers):
    print(i, layer.name, layer.trainable)

conv_base.summary()

# freeze all layers except the last 4 
for layer in conv_base.layers[:15]:
    layer.trainable = False

for i, layer in enumerate(conv_base.layers):
    print(i, layer.name, layer.trainable)

conv_base.summary()

# print the trainable status of individual layers
for layer in conv_base.layers:   print(layer,"  ",  layer.trainable)

# number of epoch we used to train the classifier earlier
initial_epochs   = 5

# fine-tune the model for 10 epochs (in addition to previous 5 epochs)
fine_tune_epochs = 10

total_epochs     =  initial_epochs + fine_tune_epochs


model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.RMSprop(learning_rate=1e-5),
                  metrics='accuracy')

if training:
    history_fine_tune = model.fit(train_ds, batch_size = 32, initial_epoch=initial_epochs, epochs=total_epochs, validation_data=val_ds)
    model.save('model_fine_tune.h5')
else:
    model = tf.keras.models.load_model('model_fine_tune.h5')

history.history['accuracy'] += history_fine_tune.history['accuracy']
history.history['val_accuracy'] += history_fine_tune.history['val_accuracy']

history.history['loss'] += history_fine_tune.history['loss']
history.history['val_loss'] += history_fine_tune.history['val_loss']

acc      = history.history['accuracy']
val_acc  = history.history['val_accuracy']
loss     = history.history['loss']
val_loss = history.history['val_loss']

fig, ax = plt.subplots(1,2, figsize=(10,3))
df_accuracy = pd.DataFrame(history.history).loc[:,['accuracy','val_accuracy']]
df_loss = pd.DataFrame(history.history).loc[:,['loss','val_loss']]

df_accuracy.plot(ax=ax[0])
df_loss.plot(ax=ax[1])
ax[0].set_ylim(0.5,1.05)
ax[1].set_ylim(-0.5,5)

plt.show()

