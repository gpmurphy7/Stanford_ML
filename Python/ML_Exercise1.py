
# coding: utf-8

# # Excercise 1

# ## Part 1

# In[1]:

#importing libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[2]:

#loading in data "ex1data1.txt" from the data folder
path = os.getcwd() + '\data\ex1data1.txt'
data = pd.read_csv(path, header = None, names = ['Population', 'Profit'])
data.head()


# In[3]:

#explore data with describe
data.describe()


# In[4]:

#plotting the data in a scatter plot
data.plot(kind = 'scatter', x = 'Population', y = 'Profit')


# In[5]:

#compute a cost function to evaluate model error and quality
#least squares method.
def computeCost(X,y,theta):
    inner = np.power(((X*theta.T)-y), 2)
    return np.sum(inner) / (2*len(X))


# In[6]:

# append column of ones at front of data set
data.insert(0, 'Ones', 1)

# set X(training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]
theta = np.matrix(np.array([0,0]))


# In[19]:

#convert to numpy matrices
X = np.matrix(X)
y = np.matrix(y)


# In[20]:

X.shape,theta.shape,y.shape


# In[21]:

computeCost(X,y,theta)


# In[22]:

def gradientDescent(X, y, theta, alpha, iters):  
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(X, y, theta)

    return theta, cost


# In[23]:

# initialize variables for learning rate and iterations
alpha = 0.01  
iters = 1000


# In[24]:

# perform gradient descent to "fit" the model parameters
g, cost = gradientDescent(X, y, theta, alpha, iters)  
g  


# In[25]:

computeCost(X,y,g)


# In[50]:

#visualisation

x = np.linspace(data.Population.min(), data.Population.max(), 100)  
f = g[0, 0] + (g[0, 1] * x)

fig, ax = plt.subplots()  
ax.plot(x, f, 'r', label='Prediction')  
ax.scatter(data.Population, data.Profit, label='Traning Data')  
ax.legend(loc=2)  
ax.set_xlabel('Population')  
ax.set_ylabel('Profit')  
ax.set_title('Predicted Profit vs. Population Size') 


# In[51]:

fig, ax = plt.subplots()  
ax.plot(np.arange(iters), cost, 'r')  
ax.set_xlabel('Iterations')  
ax.set_ylabel('Cost')  
ax.set_title('Error vs. Training Epoch')  


# ## Part 2

# In[52]:

path2 = os.getcwd() + '\data\ex1data2.txt'
data2 = pd.read_csv(path2, header = None, names=['Size', 'Bedrooms', 'Price'])
data2.head()


# In[53]:

data2 = (data2 - data2.mean()) / data2.std()
data2.head()


# In[54]:

# add ones column
data2.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data2.shape[1]
X2 = data2.iloc[:,0:cols-1]
y2 = data2.iloc[:,cols-1:cols]


# In[55]:

X2 = np.matrix(X2)
y2 = np.matrix(y2)
theta2 = np.matrix(np.array([0,0,0]))


# In[56]:

# perform linear regression on the data set
g2, cost2 = gradientDescent(X2, y2, theta2, alpha, iters)

# get the cost (error) of the model
computeCost(X2, y2, g2)  


# In[57]:

fig, ax = plt.subplots()  
ax.plot(np.arange(iters), cost2, 'r')  
ax.set_xlabel('Iterations')  
ax.set_ylabel('Cost')  
ax.set_title('Error vs. Training Epoch') 


# In[58]:

from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(X,y)


# In[61]:

x = np.array(X[:, 1].A1)  
f = model.predict(X).flatten()

fig, ax = plt.subplots()  
ax.plot(x, f, 'r', label='Prediction')  
ax.scatter(data.Population, data.Profit, label='Traning Data')  
ax.legend(loc=2)  
ax.set_xlabel('Population')  
ax.set_ylabel('Profit')  
ax.set_title('Predicted Profit vs. Population Size')


# In[ ]:



