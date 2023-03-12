# The mathematics of optimization for Deep Learning

Mathematically a neural network is a composition of affine mappings, and non-linear (activation) mappings, where we alternate between an affine mapping (weights and biases) and an activation function (applied element wise).

**Optimization Algorithms**

For any 𝑓(𝑥) function, if the value of 𝑓(𝑥) at 𝑥 is smaller than the values of 𝑓(𝑥) at any other points in the proximity of 𝑥, then 𝑓(𝑥) could be a local minimum. 

If the value of  𝑓(𝑥) at 𝑥 is the minimum of the 𝑓(𝑥) function over the entire domain, then 𝑓(𝑥) is the global minimum.

We can approximate the local minimum and global minimum of the following 𝑓(𝑥)function.

![alt text](https://github.com/jylhakos/Deep-Learning-with-Python/blob/main/Gradient-Based-Learning/function.png?raw=true)

![alt text](https://github.com/jylhakos/Deep-Learning-with-Python/blob/main/Gradient-Based-Learning/optimization.svg?raw=true)


## Gradient Based Optimization

Gradient Descent (GD) minimizes the training error by incrementally improving the current guess for the optimal parameters by moving a bit into the direction of the negative gradient.

Gradient Descent is used to tune (adjust) the parameters according to the gradient of the average loss incurred by the Artificial Neural Network (ANN) on a training set. 

This average loss is also known as the training error and defines a cost function 𝑓(𝐰) that we want to minimize.

For a given pair of predicted label value ŷ and true label value 𝑦, the loss function 𝐿(𝑦,ŷ) provides a measure for the error, or "loss", incurred in predicting the true label 𝑦 by ŷ.

If the label values are numeric (like a temperature), then the squared error loss 𝐿(𝑦,ŷ)=(𝑦−ŷ)² is often a good choice for the loss function. If the label values are categories (like "cat" and "dog"), we might use the "0/1" loss 𝐿(𝑦,ŷ)=0  if and only if 𝑦=ŷ and 𝐿(𝑦,ŷ)=1 otherwise.

Gradient descent (GD) constructs a sequence of parameter vectors 𝐰(0),𝐰(1),... such that the loss values 𝑓(𝐰(0)),𝑓(𝐰(1)),... pass toward the minimum loss. GD is an iterative algorithm that gradually improves the current guess (approximation) 𝐰(𝑘) for the optimum weight vector.

```
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

def gradient_step_onefeature(x,y,weight,lrate):
	y_hat = x.dot(weight).flatten()
    print('y_hat', y_hat)
    
    # 2. compute MSE loss
    error = y.flatten() - y_hat

    m = len(y)

    MSE = ((1.0 / m) * (np.sum(np.power(error,2))))
    
    print('MSE', MSE)
    
    # 3. compute the average gradient of the loss function
    grad_w = (-2/m)*(error.dot(x))
    
    print('grad_w', grad_w)
    
    # 4. update the weights
    weight = (weight - (lrate * grad_w))[0]
    
    print('weight', weight)

    return weight, MSE

# Test
from round02 import test_gradient_step_one_feature

test_gradient_step_one_feature(gradient_step_onefeature)
```
**Gradient Descent algorithm**

```
def GD_onefeature(x,y,epochs,lrate):
	# Initialize weight vector randomly
    np.random.seed(42)
    weight = np.random.rand()

    # Create a list to store the loss values 
    loss = []
    weights = []
     
    for i in range(epochs):
        # Run the gradient step for the whole data set
        weight, MSE = gradient_step_onefeature(x,y,weight,lrate)

        # Store current weight and training loss
        weights.append(weight)
        loss.append(MSE)
                 
    return weights, loss

# Generate dataset for regression problem

x, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42) 
y = y.reshape(-1,1)
x = preprocessing.scale(x)

# Set epoches and learning rate
epochs = 100
lrate = 0.1

# Store results
(weights, loss) = GD_onefeature(x,y,epochs,lrate)

# Plot loss and weight values
fig, ax = plt.subplots(1,2, figsize=(10,3), sharey=True)

# Loss vs weights plot
ax[0].plot(weights, loss)
ax[0].set_xlabel("weight", fontsize=16)
ax[0].set_ylabel("Loss", fontsize=16)

# Loss vs epoch plot
ax[1].plot(range(epochs), loss)
ax[1].set_xlabel("epoch", fontsize=16)

plt.show()

```

![alt text](https://github.com/jylhakos/Deep-Learning-with-Python/blob/main/Gradient-Based-Learning/loss_weight.png?raw=true)

**Learning Rate**

***What is the Learning Rate in neural networks?***

The Learning Rate (LR) is a hyperparameter that controls how quickly the model is adapted to the estimated error each time the model weights are updated.

Smaller Learning Rates require more training epochs given the smaller changes made to the weights each update, whereas larger learning rates result in rapid changes and require fewer training epochs.

In back-propagation, model weights are updated to reduce the error estimates of our loss function.

***Learning Rate Scheduling***

Constant Learning Rate: We initialize a learning rate and don’t change it during training.

Learning Rate decay: We select an initial learning rate, then gradually reduce it in accordance with a scheduler.


```
epochs = 100
lrates = [0.001, 0.01, 0.1, 0.9]

fig = plt.figure(figsize=(6,4))
weights_list = []
loss_list = []

for lrate in lrates:
    weight, loss = GD_onefeature(x, y, epochs, lrate)
    print('lrate', lrate, 'weight', weight, 'loss', loss)
    weights_list.append(weight)
    loss_list.append(loss)

# Plot results
for i,lrate in enumerate(lrates):
    plt.plot(weights_list[i], loss_list[i], label=f"lrate{lrate}")
    plt.legend()

plt.xlabel("weight", fontsize=16)
plt.ylabel("Loss", fontsize=16)  
plt.show()

```

![alt text](https://github.com/jylhakos/Deep-Learning-with-Python/blob/main/Gradient-Based-Learning/learning_rate.png?raw=true)

**Gradient step**

The inputs to the function are:
    - numpy array (matrix) with feature values X of shape (m,n)
    - numpy array with labels y of shape (m,1)
    - numpy array `weight` of shape (n,1), which is the weight used for computing prediction
    - scalar value `lrate`, which is a coefficient alpha used during weight update

    The function will return a new weight guess (updated weight value) and current MSE value. 

```
def gradient_step(X, y, weight, lrate):

    # 1. Compute predictions, given the feature matrix X of shape (m,n) and weight vector w of shape (n,1), while predictions should be stored in an array `y_hat` of shape (m,1).

    y_hat = (X @ weight).flatten()
    
    # 2. Compute MSE loss
    
    error = y.flatten() - y_hat
    
    m = len(y)
    
    MSE = ((1.0 / m) * (np.sum(np.power(error,2))))
    
    # 3. Compute average gradient of loss function
    gradient = ((-2/m*X.T) @ error)
    
    # 4. Update the weights

    for i in range(len(weight)):
        weight[i] = (weight[i] - (lrate * gradient[i]))
    
    return weight, MSE

# Test 
from round02 import test_gradient_step

test_gradient_step(gradient_step)
```
![alt text](https://github.com/jylhakos/Deep-Learning-with-Python/blob/main/Gradient-Based-Learning/vectorized_gradient_descent.png?raw=true)


**Stochastic Gradient Descent**

Deep learning neural networks are trained using the Stochastic Gradient Descent (SGD) algorithm.

Stochastic Gradient Descent is an optimization algorithm that estimates the error gradient for the current state of the model using examples from the training dataset, then updates the weights of the model using the back-propagation of errors algorithm.

```
def batch(X,y,batch_size):

	# Creating mini-batches of the dataset.
   
    # Check if the number of data points is equal in feature matrix X and label vector y
    
    np.random.seed(42)
    p = np.random.permutation(len(y))
    X_perm = X[p]
    y_perm = y[p]
    
    # Generate batches
    for i in range(0,X.shape[0],batch_size):
        yield (X_perm[i:i + batch_size], y_perm[i:i + batch_size])

def minibatchSGD(X, y, batch_size, epochs, lrate):  
    
    # Initialize the weight randomly
    np.random.seed(42)
    weight = np.random.rand()  
    
    # Create a list to store the loss values 
    loss = []
    weights = []
     
    for i in range(epochs):

        # Use another for-loop to iterate batch() generator and access batches one-by-one
        for mini_batch in batch(X,y,batch_size):

            X_batch, y_batch = mini_batch

            # Feed current batch to `gradient_step_onefeature()` and get weight and loss values
            weight, MSE = gradient_step_onefeature(X_batch,y_batch,weight,lrate)

            print('weight', weight, 'MSE',MSE)

            # Store current weight and loss values in corresponding lists
            weights.append(weight)
            
            loss.append(MSE)

    	# One epoch is finished when the algorithm goes through all batches
  
    return weights, loss

weights, loss = minibatchSGD(x, y, 50, 2, 0.1)
print('weights', weights, 'loss', loss)

# Set epoches and learning rate
epochs = 100
lrate = 0.02

# Iterate through the values of `batch_sizes` param
batch_sizes = [1, 10, 100]

# List for storing weights and loss for each batch size (length of both lists=3)
weights_batches = []
loss_batches = []

for batch_size in batch_sizes:
    weights, loss = minibatchSGD(x, y, batch_size, epochs, lrate)
    
    weights_batches.append(weights)
    
    loss_batches.append(loss)

    print('batch_size', batch_size, 'weights_batches', weights_batches, 'loss_batches', 

    loss_batches)
    
    print('batch_size', batch_size)

for batch_size, weights, loss in zip(batch_sizes, weights_batches, loss_batches):
    plt.plot(weights, loss, label="batch size"+str(batch_size))
    plt.legend()

plt.xlabel("weight", fontsize=16)
plt.ylabel("Loss", fontsize=16)
plt.rcParams['figure.figsize'] = [40, 20]
plt.show()

# History of the MSE loss obtained during learning
batch_size1_loss   = loss_batches[0]
batch_size10_loss  = loss_batches[1]
batch_size100_loss = loss_batches[2]

# Create the figure and axes objects
fig, axes = plt.subplots(1,3, sharey=True, figsize=(15,5))

# Create lists of loss values and batch sizes for further iteration in for-loop
batch_loss_list = [batch_size1_loss, batch_size10_loss, batch_size100_loss]
batch_size      = [1,10,100] 

for ax, batch_loss, size in zip(axes, batch_loss_list, batch_size):
    # Plot only first 100 values
    ax.plot(np.arange(len(batch_loss[:100])), batch_loss[:100])
    # Remove top and right subplot's frames 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Set subplot's title
    ax.set_title("batch size = "+str(size), fontsize=18)

# Set x- and y-axis labels
axes[0].set_xlabel('batch #', fontsize=18)
axes[0].set_ylabel('Loss', fontsize=18)

plt.ylim(0,10000)
plt.show()
```

![alt text](https://github.com/jylhakos/Deep-Learning-with-Python/blob/main/Gradient-Based-Learning/sgd_batch_size.png?raw=true)


***References***

Optimization Algorithms https://d2l.ai/chapter_optimization/index.html