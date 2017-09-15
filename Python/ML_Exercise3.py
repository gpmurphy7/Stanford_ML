
# coding: utf-8

# ### Part 4

# In[101]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
get_ipython().magic('matplotlib inline')


# In[102]:

data = loadmat("data/ex3data1.mat")


# In[103]:

data


# In[104]:

data['X'].shape, data['y'].shape


# In[105]:

def sigmoid(z):  
    return 1 / (1 + np.exp(-z))


# In[106]:

def cost(theta, X, y, learningRate):  
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / 2 * len(X)) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))
    return np.sum(first - second) / (len(X)) + reg


# In[107]:

def gradient(theta, X, y, learningRate):  
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    error = sigmoid(X * theta.T) - y

    grad = ((X.T * error) / len(X)).T + ((learningRate / len(X)) * theta)

    # intercept gradient is not regularized
    grad[0, 0] = np.sum(np.multiply(error, X[:,0])) / len(X)

    return np.array(grad).ravel()


# In[108]:

from scipy.optimize import minimize

def one_vs_all(X, y, num_labels, learning_rate):  
    rows = X.shape[0]
    params = X.shape[1]

    # k X (n + 1) array for the parameters of each of the k classifiers
    all_theta = np.zeros((num_labels, params + 1))

    # insert a column of ones at the beginning for the intercept term
    X = np.insert(X, 0, values=np.ones(rows), axis=1)

    # labels are 1-indexed instead of 0-indexed
    for i in range(1, num_labels + 1):
        theta = np.zeros(params + 1)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (rows, 1))

        # minimize the objective function
        fmin = minimize(fun=cost, x0=theta, args=(X, y_i, learning_rate), method='TNC', jac=gradient)
        all_theta[i-1,:] = fmin.x

    return all_theta


# In[109]:

rows = data['X'].shape[0]
params = data['X'].shape[1]


# In[110]:

rows


# In[111]:

all_theta = np.zeros((10, params + 1))


# In[112]:

X = np.insert(data['X'], 0, values=np.ones(rows), axis=1)


# In[113]:

theta = np.zeros(params + 1)


# In[114]:

y_0 = np.array([1 if label == 0 else 0 for label in data['y']])  
y_0 = np.reshape(y_0, (rows, 1))


# In[115]:

X.shape, y_0.shape, theta.shape, all_theta.shape


# In[116]:

np.unique(data['y'])


# In[117]:

all_theta = one_vs_all(data['X'], data['y'], 10, 1)
all_theta


# In[118]:

def predict_all(X, all_theta):  
    rows = X.shape[0]
    params = X.shape[1]
    num_labels = all_theta.shape[0]

    # same as before, insert ones to match the shape
    X = np.insert(X, 0, values=np.ones(rows), axis=1)

    # convert to matrices
    X = np.matrix(X)
    all_theta = np.matrix(all_theta)

    # compute the class probability for each class on each training instance
    h = sigmoid(X * all_theta.T)

    # create array of the index with the maximum probability
    h_argmax = np.argmax(h, axis=1)

    # because our array was zero-indexed we need to add one for the true label prediction
    h_argmax = h_argmax + 1

    return h_argmax


# In[119]:

y_pred = predict_all(data['X'], all_theta)  
correct = [1 if a == b else 0 for (a, b) in zip(y_pred, data['y'])]  
accuracy = (sum(map(int, correct)) / float(len(correct)))  
print('accuracy = {0}%'.format(accuracy * 100))


# Better in online tutorial...and better with python 2.7

# In[ ]:

# Testing sklearn


# In[55]:

from sklearn import linear_model


# In[57]:

logistic_classifier = linear_model.LogisticRegression(C=100.0)


# In[ ]:




# In[59]:

logistic_classifier.fit(data['X'], data['y'])


# In[61]:

logistic_classifier.score(data['X'], data['y'])


# In[ ]:



