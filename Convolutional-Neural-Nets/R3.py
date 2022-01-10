import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from PIL import Image
from scipy.signal import convolve2d
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import fashion_mnist
from sklearn.metrics import classification_report
import pandas as pd
from numpy.random import seed
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
        #imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        #imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
        #print(imagePadded)
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

def conv_step(matrix_slice, kernel, b):
    S = np.multiply(matrix_slice, kernel)
    Z = np.sum(S)
    Z = Z + b
    return Z

def plot_images(data_index):
    '''
        This is a function to plot first 9 images.    
        data_index: indices of images.
    
    '''
    # plot the sample images 
    f, ax = plt.subplots(3,3, figsize=(7,7))

    for i, indx in enumerate(data_index[:9]):
        ax[i//3, i%3].imshow(X_test[indx].reshape(28,28), cmap='gray')
        ax[i//3, i%3].axis('off')
        ax[i//3, i%3].set_title("True:{}  Pred:{}".format(class_names[y_test[indx]],class_names[predicted_classes[indx]]), fontsize=8)
    plt.show()    

img = mpimg.imread('R3/st1.png')
imgplot = plt.imshow(img)
plt.show()

input_matrix = np.array([[1, -1, 0], [2, 4, -1], [6, 0, 1]])
kernel = np.array([[1, 0], [0.5, -1]])
b = 0

matrix_slice = input_matrix[0:2,0:2]
#print('1. matrix_slice', matrix_slice)
x1 = conv_step(matrix_slice, kernel, b)
print("x1=", x1)

matrix_slice = np.array(input_matrix[0:2,1:3])
#print('2. matrix_slice', matrix_slice)
x2 = conv_step(matrix_slice, kernel, b)
print("x2=", x2)

matrix_slice = np.array(input_matrix[1:3,0:2])
#print('3. matrix_slice', matrix_slice)
x3 = conv_step(matrix_slice, kernel, b)
print("x3=", x3)

matrix_slice = np.array(input_matrix[1:3,1:3])
#print('4. matrix_slice', matrix_slice)
x4 = conv_step(matrix_slice, kernel, b)
print("x4=", x4)

#output_matrix = np.array([[x1,x2],[x3,x4]])
#print(output_matrix)

image = np.array([[1,-1,0], [2,4,-1], [6,0,1]])
kernel = np.array([[1,0], [0.5,-1]])
bias = 0

window = image[0:2,0:2]
#print('1. stride', stride)
x1 = convolution_operation(window, kernel, bias)
x1_cc = cross_correlation(window, kernel)
print("x1 =", x1, x1_cc[0][0])

window = np.array(image[0:2,1:3])
#print('2. window', window)
x2 = convolution_operation(window, kernel, bias)
x2_cc = cross_correlation(window, kernel)
print("x2 =", x2, x2_cc[0][0])

window = np.array(image[1:3,0:2])
#print('3. window', window)
x3 = convolution_operation(window, kernel, bias)
x3_cc = cross_correlation(window, kernel)
print("x3 =", x3, x3_cc[0][0])

window = np.array(image[1:3,1:3])
#print('4. window', window)
x4 = convolution_operation(window, kernel, bias)
x4_cc = cross_correlation(window, kernel)
print("x4 =", x4, x4_cc[0][0])


fname = Path('..') / '..' /'..' /'coursedata' / 'R3' / 'guitar.png' # file name
image = Image.open(str(fname)).convert("L") # open image with python Image library
arr = np.asarray(image) # convert image to array

# define kernel values
kernel = np.array([[-1, -1, -1],
                   [-1, 8, -1],
                   [-1, -1, -1]])

# perform convolution operation
conv_im1 = convolve2d(arr, kernel, mode='valid')

fig, axes = plt.subplots(1,3, figsize=(12,6))

axes[0].imshow(arr, cmap='gray')
axes[1].imshow(kernel, cmap='gray')
axes[2].imshow(conv_im1, cmap='gray', vmin=0, vmax=50)

axes[0].set_title('image', fontsize=20)
axes[1].set_title('kernel', fontsize=20)
axes[2].set_title('convolution', fontsize=20)

[ax.axis("off") for ax in axes]

plt.show()

np.random.seed(1)
X = np.random.randn(28, 28)
print('X.shape', X.shape)
#X = np.zeros((28, 28))
K = np.random.randn(3,3)
#K = np.zeros((3, 3))
#P = 1
#S = 1
#X_P = padding(X,P)
#print('X_P.shape', X_P.shape)
#Y = cross_correlation(X_P, K)
#print('Y.shape', Y.shape)

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

# X = 28
# K = 3
# P = (F - 1) / 2
# S = 1
# [(X - K + 2P) / S] + 1)
# 28 = (28 - 3 + 2*1) / 1 + 1
# 26 = (28 - 3 + 2*0) / 1 + 1
# 24 = (24 - 3 + 2*0) / 1 + 1

pad_cv1, stride_cv1 = 1,1
pad_cv2, stride_cv2 = 0,1
pad_cv3, stride_cv3 = 0,1

output_1 = 28-3+1
output_2 = output_1-3+1
output_3 = output_2-3+1
print('Outputs of convolutional layers', output_1, output_2, output_3)
params_conv_num = [((3*3*16) + 1) * output_1] + [((3*3*32) + 1) * output_2] + [((3*3*64) + 1) * output_3]
print('Number of convolutional layer parameters are ', params_conv_num)
params_num = (((3*3*16) + 1) * 26) + (((3*3*32) + 1) * 24) + (((3*3*64) + 1) * 22 )
print("Total number of parameters is", params_num)

seed(1)
tf.random.set_seed(1)

# load dataset
(X_trainval, y_trainval), (X_test, y_test) = fashion_mnist.load_data()

print("X Train {} and X Test size {}".format(X_trainval.shape[0], X_test.shape[0]))

# split trainval set into training and validation datasets (X_train,y_train) & (X_val, y_val) normalize feature values by 255

X_train = X_trainval[0:10000]

y_train = y_trainval[0:10000]

X_val = X_test[:6000]

print('X_test.shape',X_test.shape)

print('X_val.shape',X_val.shape)

y_val = y_test[:6000]

print('y_test.shape',y_test.shape)

print('y_val.shape', y_val.shape)

# normalize test set by 255

X_test = X_test / 255

X_test.shape

# reshape features to specify number of channels (one for grayscale images and three for RGB) as last dimension: (n_samples,height,width,channels)

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

X_val   = X_val.reshape(X_val.shape[0], 28, 28, 1)

X_test  = X_test.reshape(X_test.shape[0], 28, 28, 1)

# shape of train, validation and test datasets
print(f'Number of training examples: {X_train.shape}')
print(f'Number of validation examples: {X_val.shape}')
print(f'Number of test examples: {X_test.shape}')


model = keras.Sequential([
    layers.Conv2D(filters=16, kernel_size=3, input_shape=(28, 28, 1), padding="same", activation="relu", name="cv1"),
    layers.Conv2D(filters=16, kernel_size=3, padding="same", activation="relu", name="cv2"),
    layers.MaxPooling2D(pool_size=2, strides=(1, 1), name="maxpool"),
    layers.Flatten(name="flatten"),
    layers.Dense(64, activation="relu", name="dense"),
    layers.Dense(10, activation='softmax', name="output")
])

model.summary()

# plot graph
tf.keras.utils.plot_model(
    model, 
    show_shapes=True, 
    show_layer_names=True
)

model.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer=keras.optimizers.RMSprop(), metrics=['accuracy'])

training=True

%%time
if training:
    history = model.fit(X_train, y_train, batch_size=32,epochs=20,verbose=1,validation_data=(X_val, y_val))
    model.save('model.h5')
else: 
    model = tf.keras.models.load_model("model.h5")

if training:
    pd.DataFrame(history.history).plot(figsize=(6,4))
    plt.grid(True)
    plt.show()

# define input layer
in_layer = tf.keras.layers.Input(shape=X_train.shape[1:])

# define hidden layers and its inputs
hidden_1 = tf.keras.layers.Dense(32, activation='relu')(in_layer)
hidden_2 = tf.keras.layers.Dense(32, activation='relu')(hidden_1)

# combine two inputs
concat   = tf.keras.layers.Concatenate()([in_layer, hidden_2])

# define output layer
out      = tf.keras.layers.Dense(1, activation='relu')(concat) 

# define model with functional API
model_Func_API = tf.keras.Model(inputs=[in_layer], outputs=[out])

# plot graph
tf.keras.utils.plot_model(
    model_Func_API, 
    show_shapes=True, 
    show_layer_names=True
    )

# input layer
in_layer = model.input

# all other layers
layers = [layer.output for layer in model.layers]

# create a model 
activation_model = tf.keras.models.Model(inputs = in_layer, outputs = layers)

# this is the image whose feature map we will visualize
plt.imshow(X_test[0], cmap='gray')
plt.show()

# pass input and get feature maps of the image
activation = activation_model(X_test[0].reshape(1, 28, 28, 1))

first_layer_activation = activation[0] 
print(first_layer_activation.shape)

plt.figure(figsize=(16,16))

for i in range(first_layer_activation.shape[-1]):
    plt.subplot(8,8,i+1)
    plt.axis('off') # remove ticks
    plt.imshow(first_layer_activation[0, :, :, i], cmap='gray')
    plt.title('act. map '+ str(i+1))

plt.show()

#get the predictions for the test data
predicted_classes  = np.argmax(model(X_test.reshape(-1,28,28,1)), axis=-1)

#get true test_label
y_true = y_test

#to get the total correct and incorrect prediction from the predict class
correct=np.nonzero(predicted_classes==y_true)[0]
correct.shape[0]
incorrect=np.nonzero(predicted_classes!=y_true)[0]

print("Correct predicted classes:",correct.shape[0])
print("Incorrect predicted classes:",incorrect.shape[0])

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal',      'Shirt',   'Sneaker',  'Bag',   'Ankle boot']

target_names = [f"Class {i} ({class_names[i]}) :" for i in range(10)]
print(classification_report(y_true, predicted_classes, target_names=target_names))

# display correctly classified images
plot_images(correct)
