# Artificial-Neural-Networks.py

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# input/feature vector
x = np.array([0.14, 0.2, -5]).reshape(3,1)
# bias
b = 0.5
# weight vector
w = np.array([10, 5, 0.3]).reshape(3,1)

# compute weighted sum
# 𝑧=𝑏+𝑤1𝑥1+𝑤2𝑥2+𝑤3𝑥3
z = b + (w[0]*x[0]) + (w[1]*x[1]) + (w[2]*x[2])

# apply activation function to weighted sum z
# 𝑔(𝑧)=1/(1+𝑒−𝑧)
def sigmoid(z):
    g = 1/(1+np.exp(-z))
    g_grad = g*(1-g)
    return g, g_grad

g = 1/(1+np.exp(-z))

# print the results
print("The output is: ", g, "Shape", g.shape)

# input - feature values of 100 data points
x = np.linspace(-10, 10, 100).reshape(100,1)

# weight and bias of first hidden neuron
w11, b11 = -1, -2
# weight and bias of second hidden neuron
w12, b12 = 1, -2

# weights and bias of output neuron
b21 = 0.5
w21, w22 = 5, 3

# compute weighted sum for two hidden neurons
z1 = x.dot(w11) + b11
#print('z1', z1.shape, z1)
z2 = x.dot(w12) + b12
#print('z2', z2.shape, z2)
# compute weighted sum of hidden neurons' outputs (without activation)
h = z1.dot(w21) + z2.dot(w22) + b21
print('h[0]', h[0], 'h[42]', h[42], 'h.shape', h.shape)
# plot outputs of neurons
fig, axes = plt.subplots(1,3, sharey=True, figsize=(8,2))

# output of first hidden neuron
axes[0].plot(x, z1)
# output of second hidden neuron
axes[1].plot(x, z2)
# output of output neuron
axes[2].plot(x, h)

axes[0].set_title('$z_{1} = b^{(1)}_{1} + w^{(1)}_{1}x$', fontsize=12)
axes[1].set_title('$z_{1} = b^{(1)}_{2} + w^{(1)}_{2}x$', fontsize=12)
axes[2].set_title('$h(x) = b^{(2)}_{1} + w^{(2)}_{1}z_{1} + w^{(2)}_{2}z_{2}$', fontsize=12)

plt.show()

def ReLU(z):
    return np.where(z > 0, z, 0)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# use weighted sum z1 and z2 computed in previous task and 
# apply ReLU activation function to z1
g1 = ReLU(z1)
# apply ReLU activation function to z2
g2 = ReLU(z2)
# compute weighted sum of hidden neurons' outputs 
h =  b11 + w11*g1 + w12*g2

print('h.shape', h.shape, 'h[42]', h[42], 'h[0]', h[0])

g3 = sigmoid(z1)

g4 = sigmoid(z2)

h2 =  b11 + w11*g3 + w12*g4

print('h2.shape', h2.shape, 'h2[42]', h2[42], 'h2[0]', h2[0])

# plot outputs of neurons
fig, axes = plt.subplots(2,3, figsize=(7,4))

axes[0,0].plot(x, z1) # weighted sum of first hidden neuron
axes[0,1].plot(x, z2) # weighted sum of second hidden neuron
axes[0,2].axis('off') # hide axis of extra subplot

axes[1,0].plot(x, g1) # activation of first hidden neuron
axes[1,1].plot(x, g2) # activation of second hidden neuron
axes[1,2].plot(x, h)  # output 
axes[1,0].plot(x, g3) # activation of first hidden neuron
axes[1,1].plot(x, g3) # activation of second hidden neuron
axes[1,2].plot(x, h2) # output 

axes[0,0].set_title('$z_{1} = b^{(1)}_{1} + w^{(1)}_{1}x$', fontsize=12)
axes[0,1].set_title('$z_{1} = b^{(1)}_{2} + w^{(1)}_{2}x$', fontsize=12)
axes[1,0].set_title('$g(z_{1}) = max(0,z_{1})$', fontsize=12)
axes[1,1].set_title('$g(z_{2}) = max(0,z_{2})$', fontsize=12)
axes[1,2].set_title('$h(x) = b^{(2)}_{1} + w^{(2)}_{1}g(z_{1}) + w^{(2)}_{2}g(z_{2})$', fontsize=12)

fig.tight_layout()
plt.show()

# load dataset
from tensorflow.keras.datasets import fashion_mnist
(trainval_images, trainval_labels), (test_images, test_labels) = fashion_mnist.load_data()

# shape of train and test image
print(f'Number of training and validation examples {trainval_images.shape}')
print(f'Number of test examples {test_images.shape}')
print(f'Min feature value {trainval_images.min()}')
print(f'Max feature value {trainval_images.max()}')
print(f'Data type {type(trainval_images.min())}')

labels = np.unique(test_labels)
print(labels)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal',      'Shirt',   'Sneaker',  'Bag',   'Ankle boot']

# display numeric label and corresponding class name 
print('label value \t category')
for class_name, label in zip(class_names, labels):
    print (f'{label} \t\t {class_name}')

# visuale 10 first images from training set
plt.figure(figsize=(10,10))
i = 0
for (image, label) in zip(test_images[:10],test_labels[:10]):
    plt.subplot(5,5,i+1)
    plt.xticks([]) # remove ticks on x-axis
    plt.yticks([]) # remove ticks on y-axis
    plt.imshow(image, cmap='binary') # set the colormap to 'binary' 
    plt.xlabel(class_names[label])
    i += 1
plt.tight_layout()    
plt.show()

# select the image to visualize
img = test_images[0]
# create figure and axis objects
fig, ax = plt.subplots(1,1,figsize = (10,10)) 
# display image
ax.imshow(img, cmap='gray')
width, height = img.shape
# this value will be needed in order to change the color of annotations
thresh = img.max()/2.5

# display grayscale value of each pixel
for x in range(width):
    for y in range(height):
        val = (img[x][y])
        ax.annotate(str(val), xy=(y,x),
                    horizontalalignment='center',
                    verticalalignment='center',
                    # if a pixel is black set the color of annotation as white
                    color='white' if img[x][y]<thresh else 'black')
plt.show()

# select subset of trainval_images and trainval_labels
X_trainval = trainval_images[:16000]
y_trainval = trainval_labels[:16000]

# select whole test set
X_test = test_images
y_test = test_labels

X_trainval = X_trainval.reshape(-1, 28 * 28)
X_test = test_images.reshape(-1, 28 * 28)

# Normalize data to have feature values between 0 and 1
X_trainval = X_trainval / 255.0
X_test = X_test / 255.0

# method 1
# create an object 'model' that represents an ANN
model = keras.Sequential()      
# add first (input) layer and second dense layer by using `model.add()` method
model.add(layers.InputLayer(input_shape=(784,)))
model.add(layers.Dense(units=128, activation='relu'))  

# method 2
# make a list of layers and pass to `keras.Sequential()`
model = keras.Sequential([
  layers.InputLayer(input_shape=(784,)),
  layers.Dense(units=128, activation='relu')
])

# method 3
# skip input layer and indicate input shape in the first dense layer instead
model = keras.Sequential([
  layers.Dense(units=128, activation='relu',input_shape=(784,))
])

# define model architecture

model = keras.Sequential([
    # hidden layers
    layers.Dense(128, activation='relu',input_shape=(784,)),
    # output layer
    layers.Dense(10, activation='softmax')
])

model.summary()

keras.utils.plot_model(
    model,
    show_shapes=True, 
    show_layer_names=True
)

y_onehot = keras.utils.to_categorical(y_test)  

print("label in numeric form of first data point in test set: ", y_test[0])
print("label in one-hot form of first data point in test set: ", y_onehot[0])

# compile the model
model.compile(optimizer='RMSprop',
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

%%time 
# track execution time

if training==True:
    history = model.fit(X_trainval, y_trainval, validation_split=0.2, batch_size=32, epochs=20, verbose=1)

import pandas as pd

# plot training log
if training==True:
    pd.DataFrame(history.history).plot(figsize=(7,4))
    plt.grid(True)
    plt.xlabel('epoch', fontsize=14)
    plt.show()

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print('Accuracy on test dataset:', test_accuracy)


from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# function to load dataset
def load_dataset():
    
    X, y = fetch_california_housing(return_X_y=True)
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2)
    
    # scale feature values
    scaler = StandardScaler()
    X_trainval = scaler.fit_transform(X_trainval)
    X_test = scaler.transform(X_test)
    
    return X_trainval, y_trainval, X_test, y_test

# load dataset
X_reg_trainval, y_reg_trainval, X_reg_test, y_reg_test = load_dataset()

# shape of train and test image
print(f'Number of training and validation examples {X_reg_trainval.shape}')
print(f'Number of test examples {X_reg_test.shape}')

if training==True:
    model_reg = keras.Sequential()
    model_reg.add(layers.Dense(128, input_dim=8, kernel_initializer='normal', activation='relu'))
    model_reg.add(layers.Dense(1, kernel_initializer='normal'))
    # Compile model
    model_reg.compile(loss='mean_squared_error', optimizer='adam', metrics='mean_squared_error')
    history = model_reg.fit(X_reg_trainval, y_reg_trainval, epochs=20)
    model_reg.save('model_reg.h5')
else: 
    model_reg = tf.keras.models.load_model("model_reg.h5")
    
keras.utils.plot_model(
    model_reg,
    show_shapes=True, 
    show_layer_names=True
)

# plot training log
if training==True:
    pd.DataFrame(history.history).plot(figsize=(6,3))
    plt.grid(True)
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.show()

test_loss = model_reg.evaluate(X_reg_test,y_reg_test, batch_size=128, verbose=0)
print('MSE loss on test dataset:', test_loss)

%%time
#------------MODEL 1--------------------#

# build a model with one hidden layer with 256 units and ReLU activation
# model_256 = keras.Sequential([...])

# compile a model
# model_256.compile(...)

if training==True:
    # train a model
    model_256 = keras.Sequential()
    model_256.add(layers.Dense(256, input_dim=784, kernel_initializer='normal', activation='relu'))
    model_256.add(layers.Dense(10, kernel_initializer='normal', activation='softmax'))
    # Compile model
    model_256.compile(loss='mean_squared_error', optimizer='adam', metrics='accuracy')
    history_256 = model_256.fit(X_trainval, y_trainval, validation_split=0.2, batch_size=32, epochs=20)
    model_256.save('model_256.h5')
else:
    model_256 = tf.keras.models.load_model("model_256.h5")
    
# evaluate a model
test_loss, test_accuracy = model_256.evaluate(X_test, y_test, verbose=0)
print('Accuracy on test dataset:', test_accuracy)
keras.utils.plot_model(
    model_256,
    show_shapes=True, 
    show_layer_names=True
)

%%time
#------------MODEL 2--------------------#

# build a model with 4 hidden layers with 64 units and ReLU activations
# model_4x64 = keras.Sequential([...])

# compile a model
# model_4x64.compile(...)

if training==True:
    # train a model
    model_4x64 = keras.Sequential()
    model_4x64.add(layers.Dense(64, input_dim=784, kernel_initializer='normal', activation='relu'))
    model_4x64.add(layers.Dense(64, input_dim=784, kernel_initializer='normal', activation='relu'))
    model_4x64.add(layers.Dense(64, input_dim=784, kernel_initializer='normal', activation='relu'))
    model_4x64.add(layers.Dense(64, input_dim=784, kernel_initializer='normal', activation='relu'))
    model_4x64.add(layers.Dense(10, kernel_initializer='normal', activation='softmax'))
    # Compile model
    model_4x64.compile(loss='mean_squared_error', optimizer='adam', metrics='mean_squared_error')
    history_4x64 = model_4x64.fit(X_trainval, y_trainval, validation_split=0.2, batch_size=32, epochs=20)
    model_4x64.save('model_4x64.h5')
else: 
    model_4x64 = keras.models.load_model("model_4x64.h5")

# evaluate a model
test_loss, test_accuracy = model_4x64.evaluate(X_test, y_test, verbose=0)
print('Accuracy on test dataset:', test_accuracy, 'test_loss', test_loss)
keras.utils.plot_model(
    model_4x64,
    show_shapes=True, 
    show_layer_names=True
)

%%time

lrates = [0.0001, 0.001, 0.01, 0.1, 1, 10]
test_acc = []

# define function; use for-loop to iterate list values
best_learning_rate = 0.0
def lrate(lrates):

    print('lrates',lrates)
    
    best_lrate = 0.0
    
    highest_test_accuracy = 0.0

    for lr in lrates:
        optimizer = keras.optimizers.SGD(learning_rate=lr)
        model = keras.Sequential()
        model.add(layers.Dense(128, input_dim=784, kernel_initializer='normal', activation='relu'))
        model.add(layers.Dense(10, kernel_initializer='normal', activation='softmax'))
        print('model.compile')
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics='accuracy')
        print('model.fit')
        history = model.fit(X_trainval, y_trainval, validation_split=0.2, batch_size=32, epochs=20)
        print('model.evaluate')
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        test_acc.append(test_accuracy)

        if (test_accuracy > highest_test_accuracy):
            model.save('model.h5')
            best_lrate = lr
            highest_test_accuracy = test_accuracy
            print('model.save: learning_rate=', lr)

        print('test_accuracy', test_accuracy)

    return test_acc, best_lrate

if training==True:
    test_acc, best_learning_rate = lrate(lrates)
    print('test_acc', test_acc, 'best_learning_rate', best_learning_rate)
else: 
    model = keras.models.load_model("model.h5")

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print('loss', loss, 'accuracy', accuracy)

keras.utils.plot_model(
    model,
    show_shapes=True, 
    show_layer_names=True
)

# plot graph lrate vs test accuracy
if training==True:
    fig, ax = plt.subplots(1,1, sharey=True, figsize=(5,3))
    
    ax.plot(lrates, test_acc)
    plt.xlabel("learning rate", fontsize=14)
    plt.ylabel("Test accuracy", fontsize=14)
    plt.xscale('log')   
    plt.show()

