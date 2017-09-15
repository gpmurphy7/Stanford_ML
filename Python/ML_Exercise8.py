
# coding: utf-8

# ## Anomaly Detection

# Our first task is to use a Gaussian model to detect if an unlabeled example from a data set should be considered an anomaly. We have a simple 2-dimensional data set to start off with so we can easily visualize what the algorithm is doing. Let's pull in and plot the data.

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
get_ipython().magic('matplotlib inline')


# In[2]:

data = loadmat('data/ex8data1.mat')


# In[3]:

X = data['X']


# In[4]:

X.shape


# In[5]:

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(X[:,0], X[:,1])
plt.show()


# It appears that there's a pretty tight cluster in the center with several values further out away from the cluster. In this simple example, these could be considered anomalies. To find out, we're tasked with estimating a Gaussian distribution for each feature in the data. You may recall that to define a probability distribution we need two things - mean and variance. To accomplish this we'll create a simple function that calculates the mean and variance for each feature in our data set.

# In[6]:

def estimate_gaussian(X):
    mu = X.mean(axis=0)
    sigma = X.var(axis=0)
    
    return mu, sigma


# In[7]:

mu, sigma = estimate_gaussian(X)


# In[8]:

mu, sigma


# Now that we have our model parameters, we need to determine a probability threshold which indicates that an example should be considered an anomaly. To do this, we need to use a set of labeled validation data (where the true anomalies have been marked for us) and test the model's performance at identifying those anomalies given different threshold values.

# In[9]:

Xval = data['Xval']
yval = data['yval']


# In[10]:

Xval.shape, yval.shape


# We also need a way to calculate the probability that a data point belongs to a normal distribution given some set of parameters. Fortunately SciPy has this built-in.

# In[11]:

from scipy import stats


# In[12]:

dist = stats.norm(mu[0], sigma[0])


# In[13]:

dist.pdf(X[:,0])[0:50]


# In case it isn't clear, we just calculated the probability that each of the first 50 instances of our data set's first dimension belong to the distribution that we defined earlier by calculating the mean and variance for that dimension. Essentially it's computing how far each instance is from the mean and how that compares to the "typical" distance from the mean for this data.

# Let's compute and save the probability density of each of the values in our data set given the Gaussian model parameters we calculated above.

# In[14]:

p = np.zeros((X.shape[0], X.shape[1]))
p[:,0] = stats.norm(mu[0], sigma[0]).pdf(X[:,0])
p[:,1] = stats.norm(mu[1], sigma[1]).pdf(X[:,1])


# In[15]:

p.shape


# We also need to do this for the validation set (using the same model parameters). We'll use these probabilities combined with the true label to determine the optimal probability threshold to assign data points as anomalies.

# In[16]:

pval = np.zeros((X.shape[0], X.shape[1]))
pval[:,0] = stats.norm(mu[0], sigma[0]).pdf(Xval[:,0])
pval[:,1] = stats.norm(mu[1], sigma[1]).pdf(Xval[:,1])


# Next, we need a function that finds the best threshold value given the probability density values and true labels. To do this we'll calculate the F1 score for varying values of epsilon. F1 is a function of the number of true positives, false positives, and false negatives.

# In[17]:

def select_threshold(pval,yval):
    best_epsilon = 0
    best_f1 = 0
    f1 = 0
    
    step = (pval.max() - pval.min())/1000
    
    for epsilon in np.arange(pval.min(), pval.max(), step):
        preds = pval < epsilon
        
        tp = np.sum(np.logical_and(preds == 1, yval == 1)).astype(float)
        fp = np.sum(np.logical_and(preds == 1, yval == 0)).astype(float)
        fn = np.sum(np.logical_and(preds == 0, yval == 1)).astype(float)
        
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2* precision * recall) / (precision + recall)
        
        if f1 > best_f1:
            best_f1 = f1
            best_epsilon = epsilon
            
    return best_epsilon, best_f1


# In[18]:

epsilon, f1 = select_threshold(pval, yval)


# In[19]:

epsilon, f1


# Finally, we can apply the threshold to the data set and visualize the results.

# In[20]:

# indexes of values considered to be outliers
outliers = np.where(p < epsilon)

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(X[:,0], X[:,1])
ax.scatter(X[outliers[0],0], X[outliers[0],1], s=50, color='r', marker='o')
plt.show()


# The points in red are the ones that were flagged as outliers. Visually these seem pretty reasonable. The top right point that has some separation (but was not flagged) may be an outlier too, but it's fairly close. There's another example in the text of applying this to a higher-dimensional data set, but since it's a trivial extension of the two-dimensional example we'll move on to the last section.

# ## Collaborative Filtering

# Recommendation engines use item and user-based similarity measures to examine a user's historical preferences to make recommendations for new "things" the user might be interested in. In this exercise we'll implement a particular recommendation algorithm called collaborative filtering and apply it to a data set of movie ratings. Let's first load and examine the data we'll be working with.

# In[21]:

data = loadmat('data/ex8_movies.mat')
data


# Y is a (number of movies x number of users) array containing ratings from 1 to 5. R is an "indicator" array containing binary values indicating if a user has rated a movie or not. Both should have the same shape.

# In[22]:

Y = data['Y']
R = data['R']
Y.shape, R.shape


# We can look at the average rating for a movie by averaging over a row in Y for indexes where a rating is present.

# In[23]:

Y[1,R[1,:]].mean()


# Next we're going to implement a cost function for collaborative filtering. Intuitively, the "cost" is the degree to which a set of movie rating predictions deviate from the true predictions. The cost equation is given in the exercise text. It is based on two sets of parameter matrices called X and Theta in the text. These are "unrolled" into the "params" input so that we can use SciPy's optimization package later on. Note that I've included the array/matrix shapes in comments to help illustrate how the matrix interactions work.

# In[24]:

def cost(params, Y, R, num_features):
    Y = np.matrix(Y) # (1682, 943)
    R = np.matrix(R) # (1682, 943)
    num_movies = Y.shape[0]
    num_users = Y.shape[1]
    
    #reshape the parameter array into parameter matrices
    X = np.matrix(np.reshape(params[:num_movies * num_features], (num_movies, num_features))) # (1682,10)
    Theta = np.matrix(np.reshape(params[num_movies * num_features:], (num_users, num_features)))# (943, 10)
    
    #initialise
    J = 0
    
    error = np.multiply((X * Theta.T) - Y, R) # (1682, 943)
    squared_error = np.power(error, 2) # (1682, 943)
    J = (1. / 2) * np.sum(squared_error)
    
    return J


# In order to test this, we're provided with a set of pre-trained parameters that we can evaluate. To keep the evaluation time down, we'll look at just a small sub-set of the data.

# In[25]:

users = 4
movies = 5
features = 3

params_data = loadmat('data/ex8_movieParams.mat')
X = params_data['X']
Theta = params_data['Theta']

X_sub = X[:movies, :features]
Theta_sub = Theta[:users,:features]
Y_sub = Y[:movies, :users]
R_sub = R[:movies, :users]

params = np.concatenate((np.ravel(X_sub), np.ravel(Theta_sub)))

cost(params, Y_sub, R_sub, features)


# This answer matches what the exercise text said we're supposed to get. Next we need to implement the gradient computations. Just like we did with the neural networks implementation in exercise 4, we'll extend the cost function to also compute the gradients.

# In[26]:

def cost(params, Y, R, num_features):
    Y = np.matrix(Y) # (1682, 943)
    R = np.matrix(R) # (1682, 943)
    num_movies = Y.shape[0]
    num_users = Y.shape[1]
    
    #reshape the parameter array into parameter matrices
    X = np.matrix(np.reshape(params[:num_movies * num_features], (num_movies, num_features))) # (1682,10)
    Theta = np.matrix(np.reshape(params[num_movies * num_features:], (num_users, num_features)))# (943, 10)
    
    #initialise
    J = 0
    X_grad = np.zeros(X.shape) # (1682, 10)
    Theta_grad = np.zeros(Theta.shape) # (943, 10)
    
    #compute cost
    error = np.multiply((X * Theta.T) - Y, R) # (1682, 943)
    squared_error = np.power(error, 2) # (1682, 943)
    J = (1. / 2) * np.sum(squared_error)
    
    #calcualte the gradients
    X_grad = error * Theta
    Theta_grad = error.T  * X
    
    #unravel the gradient matrixes into single array
    grad = np.concatenate((np.ravel(X_grad), np.ravel(Theta_grad)))
    
    return J, grad


# In[27]:

J, grad = cost(params, Y_sub, R_sub, features)


# In[28]:

J, grad


# Our next step is to add regularization to both the cost and gradient calculations. We'll create one final regularized version of the function (note that this version includes an additional learning rate parameter called "lambda").

# In[29]:

def cost(params, Y, R, num_features, learning_rate):  
    Y = np.matrix(Y)  # (1682, 943)
    R = np.matrix(R)  # (1682, 943)
    num_movies = Y.shape[0]
    num_users = Y.shape[1]

    # reshape the parameter array into parameter matrices
    X = np.matrix(np.reshape(params[:num_movies * num_features], (num_movies, num_features)))  # (1682, 10)
    Theta = np.matrix(np.reshape(params[num_movies * num_features:], (num_users, num_features)))  # (943, 10)

    # initializations
    J = 0
    X_grad = np.zeros(X.shape)  # (1682, 10)
    Theta_grad = np.zeros(Theta.shape)  # (943, 10)

    # compute the cost
    error = np.multiply((X * Theta.T) - Y, R)  # (1682, 943)
    squared_error = np.power(error, 2)  # (1682, 943)
    J = (1. / 2) * np.sum(squared_error)

    # add the cost regularization
    J = J + ((learning_rate / 2) * np.sum(np.power(Theta, 2)))
    J = J + ((learning_rate / 2) * np.sum(np.power(X, 2)))

    # calculate the gradients with regularization
    X_grad = (error * Theta) + (learning_rate * X)
    Theta_grad = (error.T * X) + (learning_rate * Theta)

    # unravel the gradient matrices into a single array
    grad = np.concatenate((np.ravel(X_grad), np.ravel(Theta_grad)))

    return J, grad


# In[30]:

J, grad = cost(params, Y_sub, R_sub, features, 1.5) 


# In[31]:

J, grad


# This result again matches up with the expected output from the exercise code, so it looks like the regularization is working. Before we train the model, we have one final step. We're tasked with creating our own movie ratings so we can use the model to generate personalized recommendations. A file is provided for us that links the movie index to its title. Let's load the file into a dictionary and use some sample ratings provided in the exercise.

# In[32]:

movie_idx = {}
f = open('data/movie_ids.txt')
for line in f:
    tokens = line.split(' ')
    tokens[-1] = tokens[-1][:-1]
    movie_idx[int(tokens[0]) -1] = ' '.join(tokens[1:])


# In[33]:

ratings = np.zeros((1682,1))


# In[34]:

ratings[0] = 4  
ratings[6] = 3  
ratings[11] = 5  
ratings[53] = 4  
ratings[63] = 5  
ratings[65] = 3  
ratings[68] = 5  
ratings[97] = 2  
ratings[182] = 4  
ratings[225] = 5  
ratings[354] = 5


# In[35]:

print('Rated {0} with {1} stars.'.format(movie_idx[0], str(int(ratings[0]))))  
print('Rated {0} with {1} stars.'.format(movie_idx[6], str(int(ratings[6]))))  
print('Rated {0} with {1} stars.'.format(movie_idx[11], str(int(ratings[11]))))  
print('Rated {0} with {1} stars.'.format(movie_idx[53], str(int(ratings[53]))))  
print('Rated {0} with {1} stars.'.format(movie_idx[63], str(int(ratings[63]))))  
print('Rated {0} with {1} stars.'.format(movie_idx[65], str(int(ratings[65]))))  
print('Rated {0} with {1} stars.'.format(movie_idx[68], str(int(ratings[68]))))  
print('Rated {0} with {1} stars.'.format(movie_idx[97], str(int(ratings[97]))))  
print('Rated {0} with {1} stars.'.format(movie_idx[182], str(int(ratings[182]))))  
print('Rated {0} with {1} stars.'.format(movie_idx[225], str(int(ratings[225]))))  
print('Rated {0} with {1} stars.'.format(movie_idx[354], str(int(ratings[354]))))  


# We can add this custom ratings vector to the data set so it gets included in the model.

# In[36]:

R = data['R']
Y = data['Y']


# In[37]:

Y = np.append(Y, ratings, axis=1)
R = np.append(R, ratings !=0, axis=1)


# We're now ready to train the collaborative filtering model. We're going to normalize the ratings and then run the optimization routine using our cost function, parameter vector, and data matrices at inputs.

# In[38]:

from scipy.optimize import minimize


# In[52]:

movies = Y.shape[0]  
users = Y.shape[1]  
features = 10  
learning_rate = 10.

X = np.random.random(size=(movies, features))  
Theta = np.random.random(size=(users, features))  
params = np.concatenate((np.ravel(X), np.ravel(Theta)))

Ymean = np.zeros((movies, 1))  
Ynorm = np.zeros((movies, users))

for i in range(movies):  
    idx = np.where(R[i,:] == 1)[0]
    Ymean[i] = Y[i,idx].mean()
    Ynorm[i,idx] = Y[i,idx] - Ymean[i]

fmin = minimize(fun=cost, x0=params, args=(Ynorm, R, features, learning_rate),  
                method='CG', jac=True, options={'maxiter': 500})


# In[40]:

fmin


# Since everything was "unrolled" for the optimization routine to work properly, we need to reshape our matrices back to their original dimensions.

# In[41]:

X = np.matrix(np.reshape(fmin.x[:movies * features], (movies, features)))
Theta = np.matrix(np.reshape(fmin.x[movies * features:], (users, features)))


# In[42]:

X.shape, Theta.shape


# Our trained parameters are now in X and Theta. We can use these to create some recommendations for the user we added earlier.

# In[43]:

predictions = X * Theta.T
my_preds = predictions[:,-1] + Ymean
sorted_preds = np.sort(my_preds, axis=0)[::-1]
sorted_preds[:10]


# That gives us an ordered list of the top ratings, but we lost what index those ratings are for. We actually need to use argsort so we know what movie the predicted rating corresponds to.

# In[44]:

idx = np.argsort(my_preds, axis=0)[::-1]


# In[45]:

movie_idx[312]


# In[46]:

my_preds[312]


# In[47]:

print("Top 10 movie predictions:")  
for i in range(10):  
    j = int(idx[i])
    print('Predicted rating of {0} for movie {1}.'.format(str(float(my_preds[j])), movie_idx[j]))


# Something up. Will have to come back to it

# In[ ]:



