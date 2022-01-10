# Convolutional Neural Networks (CNN)

Convolutional layers learn to "detect" a specific pattern in an image.

The densely connected layers does not preserve the structure or spatial information of the image.

There are three main types of layers in CNN:

1. Convolutional layer (conv)

2. Pooling layer (pooling)

3. Fully connected (or dense) Layer (FC)

A convolutional layer performs convolution operation between the image and kernels (also called filters).

Convolution of the image is a process where the kernel is sliding across the image and computing the weighted sum of the small area (patch) of the image.

The convolutional layer has the following hyperparameters:

1. Number of kernels, K

2. Stride length, S

3. Zero padding size, P

The pooling layer reduces the number of parameters and has following hyperparameters kernel size, F and Stride length, S

The pooling layer has operations like Max pooling and Average pooling.

The outputs of a Max pooling layer are the largest values of the corresponding patch of the input.

```
import numpy as np

def padding(X, p):
    Z = np.pad(X, ((p,p),(p,p)), 'constant')
    return Z

def convolution(image, kernel, padding, strides):
    kernel = np.flipud(np.fliplr(kernel))
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    if padding != 0:
        imagePadded = np.pad(image, ((padding,padding),(padding,padding)),'constant')
    else:
        imagePadded = image

    for y in range(image.shape[1]):
        if y > image.shape[1] - yKernShape:
            break
        if y % strides == 0:
            for x in range(image.shape[0]):
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break

    return output

def cross_correlation(X, K):
    h, w = K.shape
    Y = np.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y

def convolution_operation(window, kernel, bias):
    product = np.multiply(window, kernel)
    scalar = np.sum(product)
    scalar = scalar + bias
    return scalar

image = np.array([[1,-1,0], [2,4,-1], [6,0,1]])

kernel = np.array([[1,0], [0.5,-1]])

bias = 0

print('1. stride', stride)
window = image[0:2,0:2]
x1 = convolution_operation(window, kernel, bias)
x1_cc = cross_correlation(window, kernel)
print("x1 =", x1, x1_cc[0][0])

print('2. window', window)
window = np.array(image[0:2,1:3])
x2 = convolution_operation(window, kernel, bias)
x2_cc = cross_correlation(window, kernel)
print("x2 =", x2, x2_cc[0][0])

print('3. window', window)
window = np.array(image[1:3,0:2])
x3 = convolution_operation(window, kernel, bias)
x3_cc = cross_correlation(window, kernel)
print("x3 =", x3, x3_cc[0][0])

print('4. window', window)
window = np.array(image[1:3,1:3])
x4 = convolution_operation(window, kernel, bias)
x4_cc = cross_correlation(window, kernel)
print("x4 =", x4, x4_cc[0][0])

```

**Zero Padding**

Padding is a technique in which we add zero-valued pixels around the image symmetrically.

We call several pixels (or step size) by which kernel traversed in each slide a stride.

The size of the output from convolution operation, i.e feature map, is smaller than the input image size. 

This means that we are losing some pixel values around the perimeter of the image.

To get the input-sized output, we employ zero padding.

zero padding size = (kernel size - 1) / 2

```
np.random.seed(1)

X = np.random.randn(28, 28)

print('X.shape', X.shape)

K = np.random.randn(3,3)
P = 1
S = 1

Y_1 = convolution(X,K,P,S)

print('Y_1.shape', Y_1.shape)

P = 0
S = 1

Y_2 = convolution(Y_1,K,P,S)

print('Y_2.shape', Y_2.shape)

P = 0
S = 1

Y_3 = convolution(Y_2,K,P,S)

print('Y_3.shape', Y_3.shape)

plt.rcParams['figure.figsize'] = [40, 20]
figure, plot_matrix = plt.subplots(1,3)
plot_matrix[0].imshow(Y_1)
plot_matrix[1].imshow(Y_2)
plot_matrix[2].imshow(Y_3[0:28])

# The first convolutional layer
pad_cv1, stride_cv1 = 1,1

# The second convolutional layer
pad_cv2, stride_cv2 = 0,1

# The third convolutional layer
pad_cv3, stride_cv3 = 0,1

```

**Fully-Connected layer**

The feature map from the last convolution or pooling layer is flattened into a single vector of values and fed into a fully connected layer.

After passing through the fully connected layers, the final layer uses the softmax activation function which gives the probabilities of the input belonging to a particular class.

**Number of parameters in CNN layer**

(𝑘𝑒𝑟𝑛𝑒𝑙_𝑤𝑖𝑑𝑡ℎ ∗ 𝑘𝑒𝑟𝑛𝑒𝑙_ℎ𝑒𝑖𝑔ℎ𝑡 ∗ 𝑐ℎ𝑎𝑛𝑛𝑒𝑙𝑠_𝑖𝑛 + 1 (for bias)) ∗ 𝑐ℎ𝑎𝑛𝑛𝑒𝑙𝑠_𝑜𝑢𝑡

**CNN in Keras**

```
# Load dataset
(X_trainval, y_trainval), (X_test, y_test) = fashion_mnist.load_data()

print("X Train {} and X Test size {}".format(X_trainval.shape[0], X_test.shape[0]))

# Split trainval set into training and validation datasets (X_train,y_train) & (X_val, y_val) normalize feature values by 255

X_train = X_trainval[0:10000]

y_train = y_trainval[0:10000]

X_val = X_test[:6000]

print('X_test.shape',X_test.shape)

print('X_val.shape',X_val.shape)

y_val = y_test[:6000]

print('y_test.shape',y_test.shape)

print('y_val.shape', y_val.shape)

# Normalize test set by 255

X_test = X_test / 255

X_test.shape

# Reshape features to specify number of channels (one for grayscale images and three for RGB) by last dimension: (n_samples,height,width,channels)

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

X_val   = X_val.reshape(X_val.shape[0], 28, 28, 1)

X_test  = X_test.reshape(X_test.shape[0], 28, 28, 1)

# Shape of train, validation and test datasets
print(f'Number of training examples: {X_train.shape}')
print(f'Number of validation examples: {X_val.shape}')
print(f'Number of test examples: {X_test.shape}')

from tensorflow import keras

from tensorflow.keras import layers

model = keras.Sequential([
    layers.Conv2D(filters=16, kernel_size=3, input_shape=(28, 28, 1), padding="same", activation="relu", name="cv1"),
    layers.Conv2D(filters=16, kernel_size=3, padding="same", activation="relu", name="cv2"),
    layers.MaxPooling2D(pool_size=2, name="maxpool"),
    layers.Flatten(name="flatten"),
    layers.Dense(64, activation="relu", name="dense"),
    layers.Dense(10, activation='softmax', name="output")
])

model.summary()

# Plot a graph
tf.keras.utils.plot_model(
    model, 
    show_shapes=True, 
    show_layer_names=True
)

# Compile the model
model.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer=keras.optimizers.RMSprop(), metrics=['accuracy'])

training=True

if training:
    history = model.fit(X_train, y_train, batch_size=32,epochs=20,verbose=1,validation_data=(X_val, y_val))
    model.save('model.h5')
else: 
    model = tf.keras.models.load_model("model.h5")

import pandas as pd

if training:
    # plot training log
    pd.DataFrame(history.history).plot(figsize=(6,4))
    plt.grid(True)
    plt.show()

# Input layer
in_layer = model.input

# All other layers
layers = [layer.output for layer in model.layers]

# Create a model 
activation_model = tf.keras.models.Model(inputs = in_layer, outputs = layers)

# Pass input and get feature maps of the image
activation = activation_model(X_test[0].reshape(1, 28, 28, 1))

#We can retrieve feature maps of first convolution layer:

first_layer_activation = activation[0]

print(first_layer_activation.shape)

# Accuracy prediction on test set

predicted_classes  = np.argmax(model(X_test.reshape(-1,28,28,1)), axis=-1)

y_true=y_test

correct=np.nonzero(predicted_classes==y_true)[0]

correct.shape[0]

incorrect=np.nonzero(predicted_classes!=y_true)[0]

print("Correct predicted classes:",correct.shape[0])

print("Incorrect predicted classes:",incorrect.shape[0])

```

![alt text](https://github.com/jylhakos/Deep-Learning-with-Python/blob/main/Convolutional-Neural-Nets/cnn.png?raw=true)

