# Regularization

To prevent overfitting, the best solution is to use more complete training data. 

The dataset should cover the full range of inputs that the model is expected to handle.

The simplest way to prevent overfitting is to start with a model with a small number of learnable parameters.

**Input Pipeline**

A sequence of batches that together cover the entire dataset is called an epoch.

We divide the dataset into smaller sets called batches and only store a single batch in the working memory.

After loading new batch of data, we update the ANN parameters (weights and bias) using one iteration of Stochastic Gradient Descent (SGD) variant.

**Stochastic Gradient Descent**

In SGD, we find out the gradient of the cost function of a single sample at each iteration instead of the sum of the gradient of the cost function of all the samples.

```

import numpy as np

import tensorflow as tf

from tensorflow.keras import optimizers

import pandas as pd

import os

import pathlib 

# Create tf.data.Dataset from Python generator

ds_numbers = tf.data.Dataset.from_generator(sequence_generator, output_types=tf.int32)

# Transformations:

ds_numbers = ds_numbers.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
ds_numbers = ds_numbers.shuffle(buffer_size=20, seed=42)
ds_numbers = ds_numbers.batch(batch_size=5)
ds_numbers = ds_numbers.prefetch(buffer_size=tf.data.AUTOTUNE)

# Retrieve first batch with as_numpy_iterator() & next() functions
first_batch = list(ds_numbers.as_numpy_iterator().next())

i=1

for count_batch in ds_numbers.take(3):
    print("batch", i, count_batch.numpy())
    i += 1

base_dir = pathlib.Path.cwd() / '..' / '..' /  '..' / 'coursedata' / 'cats_and_dogs_small'

# Create tf.data.Dataset objects for training, validation and test
train_ds = tf.data.Dataset.list_files(str(base_dir/'train/*/*.jpg'), shuffle=False)

val_ds   = tf.data.Dataset.list_files(str(base_dir/'validation/*/*.jpg'), shuffle=False)

test_ds  = tf.data.Dataset.list_files(str(base_dir/'test/*/*.jpg'), shuffle=False)

for file in train_ds.take(5):
    print(file.numpy())

CLASS_NAMES = ['cats', 'dogs']
IMG_SIZE = 150

def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image, channels=3) 
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = (image / 255)
    
    parts = tf.strings.split(image_path, os.path.sep)

    one_hot = parts[-2] == CLASS_NAMES

    label = tf.argmax(one_hot)

    return (image, label)


train_ds = train_ds.map(load_image, num_parallel_calls=AUTOTUNE)

val_ds = val_ds.map(load_image, num_parallel_calls=AUTOTUNE)

test_ds = test_ds.map(load_image, num_parallel_calls=AUTOTUNE)

def configure_for_performance(ds):
    ds = ds.shuffle(buffer_size=2000)
    ds = ds.batch(batch_size=32)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

train_ds = configure_for_performance(train_ds)

val_ds   = configure_for_performance(val_ds)

test_ds  = configure_for_performance(test_ds)
```

**Data Augmentation**

```
The data augmentation artificially increases the training set by creating synthetic data points.

One method to implement data augmentation is using the concept of data generators.

CLASS_NAMES = ['cats', 'dogs']
BATCH_SIZE = 32
IMG_SIZE = 150
EPOCHS = 20

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

training=True

```

**CNN training with Data Augmentation**

```
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation, RandomZoom

data_augmentation = tf.keras.Sequential(
    [
        RandomFlip("horizontal", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        RandomRotation(0.1, fill_mode='constant'),
        RandomZoom(0.1,0.1, fill_mode='constant')
    ]
)

images, _ = train_ds.as_numpy_iterator().next()

plt.figure(figsize=(8, 8))

for i in range(9):
    augmented_images = data_augmentation(images)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_images[0].numpy())
    plt.axis("off")
plt.show()

model_aug = tf.keras.models.Sequential([
    data_augmentation,
    layers.Conv2D(filters=32, kernel_size=(3,3), padding="same", activation="relu", name="cv1"),
    layers.Conv2D(filters=32, kernel_size=(3,3), padding="same", activation="relu", name="cv2"),
    layers.Conv2D(filters=32, kernel_size=(3,3), padding="same", activation="relu", name="cv3"),
    layers.MaxPooling2D(pool_size=2, name="maxpool"),
    layers.Flatten(name="flatten"),
    layers.Dense(128, activation="relu", name="dense"),
    layers.Dense(1, activation='sigmoid', name="output")
])

model_aug.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(),
                  metrics='accuracy')

if training:
    history = model_aug.fit(train_ds, batch_size=5,epochs=2,verbose=1,validation_data=val_ds)
    model.save('model_aug.h5')
else: 
    model_aug = tf.keras.models.load_model("model_aug.h5")

plot_history(history)

check_accuracy(model_aug, 0.60)

```

**Transfer Learning**

Transfer learning is a machine learning technique in which a model trained for one particular task is used as a starting point for a training model for another task. 

```
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.applications import VGG16
from tensorflow import keras
from tensorflow.keras import layers

CLASS_NAMES = ['cats', 'dogs']
BATCH_SIZE = 32
IMG_SIZE = 60

conv_base = VGG16(weights=None, include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

head = keras.Sequential()

model = tf.keras.models.Model(conv_base.input, head(conv_base.output))

model.summary()

conv_base.summary()

model = tf.keras.models.Sequential()

model.add(conv_base)

flatten = layers.Flatten(input_shape=(1,1,conv_base.output_shape[1:]), name="flatten")

model.add(flatten)

model.add(layers.Dense(128, activation="relu", name="dense"))

model.add(layers.Dense(1, activation='sigmoid', name="output"))

model.summary()

```
**Fine-tuning pre-trained model**

```

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

from tensorflow.keras.layers import AveragePooling2D

# Create Sequential object
model = tf.keras.models.Sequential()

# Add data_augmentation block
data_augmentation.summary()

model.add(data_augmentation)

conv_base.summary()

model.add(conv_base)

clf_head = tf.keras.models.Sequential([
                                      layers.Flatten(input_shape=(1, 1, 512), name="flatten"),
                                      layers.Dense(128, activation="relu", name="dense"),
                                      layers.Dense(1, activation='sigmoid', name="output")
                                      ])

clf_head.summary()

model.add(clf_head)

model.summary()

model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.RMSprop(),
                  metrics='accuracy')

if training:
    history = model.fit(train_ds, batch_size = 32, epochs=20, validation_data=val_ds)
    model.save('model.h5')

# Unfreeeze all layers
conv_base.trainable = True

for i, layer in enumerate(conv_base.layers):
    print(i, layer.name, layer.trainable)

conv_base.summary()

# Freeze all layers except the last 4 
for layer in conv_base.layers[:15]:
    layer.trainable = False

for i, layer in enumerate(conv_base.layers):
    print(i, layer.name, layer.trainable)

conv_base.summary()

initial_epochs = 5

fine_tune_epochs = 10 

total_epochs = initial_epochs + fine_tune_epochs

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

test_loss, test_acc = model.evaluate(test_ds)

print(f'The test set accuracy of model is {test_acc:.2f}')

```

![alt text](https://github.com/jylhakos/Deep-Learning-with-Python/blob/main/Regularization/transfer_learning.png?raw=true)

