# Components of Machine Learning.py

import numpy as np
import matplotlib.pyplot as plt
from utils.styles import load_styles

load_styles()

from sklearn import datasets

# import load_iris module
from sklearn.datasets import load_iris

import pandas as pd

# load data
iris_data = load_iris(return_X_y=True, as_frame=True)

print(dir(iris_data))
# print out features' names
#print("\nFeatures:", data.feature_names)
print("\nFeatures:", iris_data)

# print out classes for classification
#print("\nClassess:", data.target_names)

fname = 'load_iris'

loader = getattr(datasets, fname)()

df = pd.DataFrame(loader['data'], columns = loader['feature_names'])

df['target'] = loader['target']

df.head(2)

#print("\nFeature:", iris_data_frame["feature_names"])

# length
print(len(data))

# load data as numpy array
X, y = load_iris(return_X_y=True)
# choose first 2 features
ind = np.where((y==1) | (y==2))[0]
y = y[ind]
X = X[ind,:2]

print("Feature matrix dimensions: ", X.shape)
print("Label vector dimensions: ", y.shape)

import seaborn as sns

# set seaborn theme for plots
sns.set_theme()

# plot data
fig, axes = plt.subplots(1,3, figsize=(15,4))
# plot histogram of first feature
sns.histplot(X[y==1,0], kde=True, ax=axes[0], color='b').set_title('first feature')
sns.histplot(X[y==2,0], kde=True, ax=axes[0], color='r')
# plot histogram of second feature
sns.histplot(X[y==1,1], kde=True, ax=axes[1], color='b').set_title('second feature')
sns.histplot(X[y==2,1], kde=True, ax=axes[1], color='r')

# plot data points
sns.scatterplot(ax=axes[2], x=X[:,0],y=X[:,1], hue=y, palette=['b','r'], legend=False)

plt.xlabel('first feature')
plt.ylabel('second feature')
plt.show()

from sklearn.linear_model import LogisticRegression

# define hypothesis space / model
clf = LogisticRegression(random_state=0)

# fit logistic regression
clf.fit(X, y)
# calculate the accuracy of the predictions
y_pred = clf.predict(X)
accuracy = clf.score(X, y)
print(f"Accuracy of classification: {round(100*accuracy, 2)}%")

# get the weights of the fitted model
w = clf.coef_ 
w = w.reshape(-1)

# minimum and maximum values of features x1 and x2
x1_min, x2_min = np.min(X, axis=0)
x1_max, x2_max = np.max(X, axis=0)

# plot the decision boundary h(x) = 0
# for data with 2 features this means w1x1 + w2x2 + bias = 0 --> x2 = (-1/w2)*(w1x1+bias)
x_grid = np.linspace(x1_min, x1_max, 100)
y_boundary = (-1/w[1])*(x_grid*w[0] + clf.intercept_)

fig, axes = plt.subplots(1, 1, figsize=(5, 4))

# plot data points belonging to class 1 and 2
sns.scatterplot(ax=axes, x=X[:,0],y=X[:,1], hue=y, palette=['b','r'], s=50, legend=False)
# plot decision boundary
axes.plot(x_grid, y_boundary, color='green')
# display x- and y-axis labels
axes.set_xlabel(r'$x_{1}$')
axes.set_ylabel(r'$x_{2}$')
# display title of figure
axes.set_title('Decision boundary', fontsize=16)
# set axes limits
axes.set_xlim(x1_min-.5, x1_max+.5)
axes.set_ylim(x2_min-0.5, x2_max+0.5)
    
plt.show()

from sklearn.model_selection import train_test_split

# load data as numpy array
X, y = load_iris(return_X_y=True)
# choose first 2 features
ind = np.where((y==1) | (y==2))[0]
y = y[ind]
X = X[ind,:2]

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print("Training set dimensions: ", X_train.shape)
print("Test set dimensions: ", X_test.shape)

# fit logistic regression
clf = LogisticRegression(random_state=0).fit(X_train, y_train)

# calculate the accuracy of the predictions
train_accuracy = clf.score(X_train, y_train)
test_accuracy = clf.score(X_test, y_test)

print(f"Training accuracy of classification: {round(100*train_accuracy, 2)}%")
print(f"Test accuracy of classification: {round(100*test_accuracy, 2)}%")

from sklearn.model_selection import cross_val_score

# create a logistic regression model
clf = LogisticRegression(random_state=0)
# data splitting to train-val sets, fitting and evaluation 
# is performed "under the hood" of `cross_val_score()` function.
# output scores are accuracies on validation folds
scores = cross_val_score(clf, X, y, cv=5)

print(f"Cross-validation scores: {scores}%")
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

# traing the model
clf.fit(X, y)

# get predictions 
predict = clf.predict(X)

# plot true and predicted labels and decision boundary
fig, axes = plt.subplots(1,2, sharex=True, sharey=True,  figsize=(9,3))

# plot data points set with true lables
sns.scatterplot(ax=axes[0], x=X[:,0],y=X[:,1], hue=y, palette=['b','r'], legend=False)
# plot decision boundary
axes[0].plot(x_grid, y_boundary, color='green')
# plot data points with predicted lables
sns.scatterplot(ax=axes[1], x=X[:,0],y=X[:,1], hue=predict, palette=['b','r'], legend=False)
# plot decision boundary
axes[1].plot(x_grid, y_boundary, color='green')

#set axes limits
axes[0].set_xlim(x1_min-.5, x1_max+.5)
axes[0].set_ylim(x2_min-0.5, x2_max+0.5)

plt.show()


