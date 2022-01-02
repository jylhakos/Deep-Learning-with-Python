import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import numpy as np
import os
import matplotlib.pyplot as plt
import pathlib
import pandas as pd

from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation, RandomZoom

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


model = keras.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3,3), padding="same", activation="relu", name="cv1"),
    layers.Conv2D(filters=32, kernel_size=(3,3), padding="same", activation="relu", name="cv2"),
    layers.Conv2D(filters=32, kernel_size=(3,3), padding="same", activation="relu", name="cv3"),
    layers.MaxPooling2D(pool_size=2, name="maxpool"),
    layers.Flatten(name="flatten"),
    layers.Dense(128, activation="relu", name="dense"),
    layers.Dense(1, activation='sigmoid', name="output")
])

model.summary()

tf.keras.utils.plot_model(
    model,
    show_shapes=True,
    show_layer_names=True
)

history = model.fit(X_train, y_train, batch_size=32,epochs=20,verbose=1,validation_data=(X_val, y_val))

model.save('model.h5')

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(),
              metrics='accuracy')


if training:
    history = model.fit(train_ds, batch_size=5,epochs=2,verbose=1,validation_data=val_ds)    model.save('model.h5')
else:
    model = tf.keras.models.load_model("model.h5")

def plot_history(history):

    if training:
        fig, ax = plt.subplots(1,2, figsize=(12,5))
        df_accuracy = pd.DataFrame(history.history).loc[:,['accuracy','val_accuracy']]
        df_loss = pd.DataFrame(history.history).loc[:,['loss','val_loss']]

        df_accuracy.plot(ax=ax[0])
        df_loss.plot(ax=ax[1])

        plt.show()

def check_accuracy(model, expected_accuracy):

    test_loss, test_acc = model.evaluate(test_ds)

    print(f'The test set accuracy of model is {test_acc:.2f}')

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

image_path = str(train_dir/'cats'/'cat.4.jpg')

image, _ = load_image(image_path)

image = image.numpy()

def visualize(original, augmented):
    plt.figure(figsize=(6, 6))
    plt.subplot(1,2,1)
    plt.title('Original image')
    plt.imshow(original)
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.title('Augmented image')
    plt.imshow(augmented)
    plt.axis("off")
    plt.show()

flipped = tf.image.flip_left_right(image)

visualize(image, flipped)

saturated = tf.image.adjust_saturation(image, 3)

visualize(image, saturated)

cropped = tf.image.central_crop(image, central_fraction=0.5)

visualize(image,cropped)

rotated = tf.image.rot90(image)

visualize(image, rotated)

def load_image_aug(image_path):
    image = tf.io.read_file(image_path)    # read the image from disk
    image = tf.io.decode_jpeg(image, channels=3)   # decode jpeg  
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])    # resize

    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_crop(image, [120, 120, 3])
    image = tf.image.resize_with_pad(image, 150, 150)
    image = (image / 255.0)    # scale 

    parts = tf.strings.split(image_path, os.path.sep)    # parse the class label from the file path
    one_hot = parts[-2] == CLASS_NAMES    # select only part with class name and create boolean array
    label = tf.argmax(one_hot)    # get label as integer from boolean array

    return (image, label)

train_ds = tf.data.Dataset.list_files(str(base_dir/'train/*/*.jpg'))

with tf.device('/cpu:0'):
    train_ds = train_ds.map(load_image_aug, num_parallel_calls=AUTOTUNE)
    train_ds = configure_for_performance(train_ds)

plt.figure(figsize=(10, 10))

for images, _ in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy())
        plt.axis("off")

plt.show()

plt.rcParams['figure.figsize'] = [20, 10]

MSELinPred_image_prefetch = plt.imread("../../../coursedata/R4/bs_aug.png")

plt.imshow(MSELinPred_image_prefetch)
