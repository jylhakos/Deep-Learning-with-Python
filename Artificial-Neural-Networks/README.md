# Artificial Neural Networks (ANN)

The goal of Machine learning (ML) is to find a hypothesis that allows to predict the label of a data point based on its features. 

Mathematically, a hypothesis is a map ℎ that reads in the features 𝑥 of a data point and outputs a prediction ℎ(𝑥) of its label.

Deep learning principle is to find a hypothesis map out of a hypothesis space that minimizes prediction error on any data point.

Deep learning uses a signal-flow chart representation, referred to as an artificial neural network (ANN), to represent a hypothesis map ℎ. 

A signal-flow chart consists of interconnected elementary units that include tunable paramters, referred to as weights (𝑤) and bias terms (𝑏).

By varying the parameters of an Artificial Neural Networks (ANN) we can select different hypothesis maps from the hypothesis space. 

We train an ANN by tuning its paramters such that resulting hypothesis brings a minimum average loss on a given set of data points (the training data).

```
    𝑧 = 𝑏 + 𝑤1𝑥1 + 𝑤2𝑥2 + 𝑤3𝑥3

```
The artificial unit then applies a non-linear activation function 𝑔(⋅) to the weighted sum 𝑧. 

The final output of the artificial unit is the function value 𝑔(𝑧), referred to as the activation of the artificial unit. 

```
    𝑔(𝑧) = 1/(1+𝑒−𝑧)

```

**An artificial unit**

```
import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

import pandas as pd

from sklearn.datasets import fetch_california_housing

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

training=True

# input/feature vector
x = np.array([0.14, 0.2, -5]).reshape(3,1)

# bias
b = 0.5

# weight vector
w = np.array([10, 5, 0.3]).reshape(3,1)

# compute weighted sum 𝑧
z = b + (w[0]*x[0]) + (w[1]*x[1]) + (w[2]*x[2])

# apply activation function 𝑔(𝑧) to weighted sum 𝑧
def sigmoid(z):
    g = 1/(1+np.exp(-z))
    g_grad = g*(1-g)
    return g, g_grad

g = 1/(1+np.exp(-z))

# print the results
print("Output: ", g, "Shape:", g.shape)

```

**Loading Data**

```
# Load Fashion-MNIST dataset
from tensorflow.keras.datasets import fashion_mnist
(trainval_images, trainval_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Shape of train and test image
print(f'Number of training and validation examples {trainval_images.shape}')
print(f'Number of test examples {test_images.shape}')

# The label values are stored as integer numbers
labels = np.unique(test_labels)
print(labels)

# Maps the numeric label values to class names. 
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker','Bag',   'Ankle boot']

for class_name, label in zip(class_names, labels):
    print (f'{label} \t\t {class_name}')


# Visuale the first images from training set
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

# Select subset of trainval_images and trainval_labels
X_trainval = trainval_images[:16000]
y_trainval = trainval_labels[:16000]

# Select whole test set
X_test = test_images
y_test = test_labels

# Flattening of a two-dimensional array of pixel grayscale values into a one-dimensional vector 𝐱 to reshape feature matrices X_trainval and X_test into the shape of model 

X_trainval = X_trainval.reshape(-1, 28 * 28)

X_test = test_images.reshape(-1, 28 * 28)

# Normalize data to have feature values between 0 and 1

X_trainval = X_trainval/ 255.0

X_test = X_test/ 255.0


```

**Define Hypothesis Space**

The ANN reads in the features 𝑥𝑖 of a shop item, which are the grayscale values of the item picture.

The output of the ANN are probabilities for each of the ten different categories.

**Layers**

The first layer consists of the features and is the entry point to the ANN. 

The first layer is connected to a dense layer with 128 artificial units with the ReLU activation function. 

This hidden layer is then followed by the final output layer with 10 artificial units and a softmax activation function.

```
# Create an object model that represents an ANN
model = keras.Sequential()

# Add first (input) layer and second dense layer by using add() method

model.add(layers.InputLayer(input_shape=(784,)))

# Hidden layer
model.add(layers.Dense(units=128, activation='relu'))

# Output layer
model.add(layers.Dense(10, activation='softmax'))

model.summary()

keras.utils.plot_model(
    model,
    show_shapes=True, 
    show_layer_names=True
)

```

![alt text](https://github.com/jylhakos/Deep-Learning-with-Python/blob/main/Artificial-Neural-Networks/plot_probability_model.png?raw=true)

**Loss Function and Optimizer**

The RMSprop is a variant of gradient descent for tuning the weights of the ANN.

We use categorical_crossentropy loss function for multiclass classification.

The metric to assess the performance of the final choice for the weights is accuracy.

```
# Convert numerical labels to one-hot encoding 

y_onehot = keras.utils.to_categorical(y_test)  

print("label in numeric form of first data point in test set: ", y_test[0])

print("label in one-hot form of first data point in test set: ", y_onehot[0])

# Compile the model
model.compile(optimizer='RMSprop',
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])


```

**Training**

A validation_split is used to separate a portion of our training data into a validation dataset and evaluate the performance of our model on that validation dataset for each epoch.

```
if training==True:
    history = model.fit(X_trainval, y_trainval, validation_split=0.2, batch_size=32, epochs=20, verbose=1)


# Plot training history
if training==True:
    pd.DataFrame(history.history).plot(figsize=(7,4))
    plt.grid(True)
    plt.xlabel('epoch', fontsize=14)
    plt.show()

```

![alt text](https://github.com/jylhakos/Deep-Learning-with-Python/blob/main/Artificial-Neural-Networks/training_probabilities_history.png?raw=true)

**Evaluation**

We evaluate the performance of the predictor map represented by ANN with the final weights.

```
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print('Accuracy on test dataset:', test_accuracy)

```

**Activation Functions**

The output 𝑔(𝑧) of an artificial unit is referred to as the activation. 

```
from actfunctions import actfunctions

actfunctions()

```

The gradient is a function for demonstrating the local gradient of the relu activation function.

![alt text](https://github.com/jylhakos/Deep-Learning-with-Python/blob/main/Artificial-Neural-Networks/gradient_relu_activation.png?raw=true)


**ANN for regression**

ANN for regression should predict real numbers and not restricted to range 0-1 as in the case with predicting probabilities. 

```

# A function to load dataset
def load_dataset():
    X, y = fetch_california_housing(return_X_y=True)

    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2)
    
    # Scale feature values
    scaler = StandardScaler()

    X_trainval = scaler.fit_transform(X_trainval)

    X_test = scaler.transform(X_test)
    
    return X_trainval, y_trainval, X_test, y_test

# Load dataset
X_reg_trainval, y_reg_trainval, X_reg_test, y_reg_test = load_dataset()

# Shape of train and test image
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

```
![alt text](https://github.com/jylhakos/Deep-Learning-with-Python/blob/main/Artificial-Neural-Networks/plot_regression_model.png?raw=true)


**Training**

```
# Plot training log
if training==True:
    pd.DataFrame(history.history).plot(figsize=(6,3))
    plt.grid(True)
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.show()

```

![alt text](https://github.com/jylhakos/Deep-Learning-with-Python/blob/main/Artificial-Neural-Networks/training_regression_history.png?raw=true)


**Loss on test set**

```
test_loss = model_reg.evaluate(X_reg_test,y_reg_test, batch_size=128, verbose=0)

print('MSE loss on test dataset:', test_loss)
```

**Model parameters and hyperparameters**

The weights and biases are parameters of a model.

The hyperparameters of a model are examples of number of layers, number of units, activation function, learning rate, batch size, number of epochs and optimizer.


