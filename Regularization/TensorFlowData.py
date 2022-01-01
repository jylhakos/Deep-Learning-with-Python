# $ python3 TensorFlowData.py

import tensorflow as tf

import numpy as np

import os

import pathlib

import matplotlib.pyplot as plt 

dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])

for value in dataset:
	print(value)
	print(value.numpy())


dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
dataset = dataset.repeat(2)

for value in dataset:
    print(value.numpy())

def preprocess(x):
    return x*x

dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
dataset = dataset.map(preprocess)

for value in dataset:
    print(value.numpy())

AUTOTUNE = tf.data.experimental.AUTOTUNE

dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
dataset = dataset.map(preprocess, num_parallel_calls=AUTOTUNE)

dataset = tf.data.Dataset.from_tensor_slices(np.arange(10))

dataset = dataset.repeat(100)
dataset = dataset.map(preprocess)
dataset = dataset.shuffle(buffer_size=20, seed=42)
dataset = dataset.batch(batch_size=5)
dataset = dataset.prefetch(buffer_size=2)

for count_batch in dataset.take(3):
    print(count_batch.numpy())

# Generators with tf.data.Dataset

def sequence_generator():   
    number = 0
    while True:
        yield number
        number += 1

numbers = []

for number in sequence_generator():
    numbers.append(number)
    if number > 9:
        break

print(numbers)

a = sequence_generator()

print("The variable is type of ", type(a))

b = next(a)

print(b)

# Create tf.data.Dataset from python generator
ds_numbers = tf.data.Dataset.from_generator(sequence_generator, output_types=tf.int32)

# Transformations
ds_numbers = ds_numbers.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
ds_numbers = ds_numbers.shuffle(buffer_size=20, seed=42)
ds_numbers = ds_numbers.batch(batch_size=5)
ds_numbers = ds_numbers.prefetch(buffer_size=tf.data.AUTOTUNE)

# Retrieve first batch with as_numpy_iterator() and next() functions
first_batch = list(ds_numbers.as_numpy_iterator().next())

print(first_batch)

# Retrieve batches with a for-loop

i=1

for count_batch in ds_numbers.take(3):
    print("batch", i, count_batch.numpy())
    i+=1

file_path = './coursedata/R4/poem.txt'

# Create dataset from txt file
dataset = tf.data.TextLineDataset(file_path)

# Print samples from dataset
for line in dataset.take(5):
    print(line.numpy())

dataset = tf.data.TextLineDataset(file_path)

for line in dataset.shuffle(20).batch(5).take(2):
    print("\nbatch:\n", line.numpy())

# The path to the dataset
base_dir = pathlib.Path.cwd() / 'coursedata' / 'cats_and_dogs'

print(base_dir)

print(type(base_dir.glob('*')))

for file in base_dir.glob('*'):
    print(file)

# Count jpg files in all subdirectories of data directroy
image_count = len(list(base_dir.glob('*/*/*.jpg')))

print(f'Total number of images in the dataset: {image_count}')

# Create tf.data.Dataset objects from training, validation and test images

# Use .list_files(file_pattern) function to select files that end with .jpg in each directory

train_ds = tf.data.Dataset.list_files(str(base_dir/'train/*/*.jpg'), shuffle=False)
val_ds   = tf.data.Dataset.list_files(str(base_dir/'validation/*/*.jpg'), shuffle=False)
test_ds  = tf.data.Dataset.list_files(str(base_dir/'test/*/*.jpg'), shuffle=False)

for file in train_ds.take(5):
    print(file.numpy())

print(f'Number of images in the training set:\t {tf.data.experimental.cardinality(train_ds).numpy()}')
print(f'Number of images in the validation set:\t {tf.data.experimental.cardinality(val_ds).numpy()}')
print(f'Number of images in the test set:\t {tf.data.experimental.cardinality(test_ds).numpy()}')

# Write custom image loader and assign a label to each loaded and transformed image. 

CLASS_NAMES = ['cats', 'dogs']
IMG_SIZE = 150

def load_image(image_path):

    # Use tf.io.read_file function to read file and tf.io.decode_jpegto decode a JPEG-encoded image to a uint8 tensor.

    image = tf.io.read_file(image_path)    # read the image from disk
    image = tf.io.decode_jpeg(image, channels=3)    # decode jpeg  
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])    # resize
    image = (image / 255)    # scale 

    # Get class names ("cats" or "dogs") from file path
    parts = tf.strings.split(image_path, os.path.sep)    # parse the class label from the file path
    one_hot = parts[-2] == CLASS_NAMES    # select only part with class name and create boolean array
    label = tf.argmax(one_hot)    # get label as integer from boolean array
 
    return (image, label)

# Use the function tf.Dataset.map to apply the function load_image to each file in the datasets and Set num_parallel_calls so multiple images are processed in parallel.

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

# Retrieve one batch (features and labels) and display images.

image_batch, label_batch = train_ds.as_numpy_iterator().next()

plt.figure(figsize=(10, 10))

for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image_batch[i])
    label = label_batch[i]
    plt.title(CLASS_NAMES[label])
    plt.axis("off")

plt.show()