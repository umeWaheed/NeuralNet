# -*- coding: utf-8 -*-
"""
Created on Sat Mar  16 09:29:11 2019

@author: ACER
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
import math

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

#number of training test samples and dimension of an image
m_train = train_set_x_orig.shape[0]     #index of the array
m_test = test_set_x_orig.shape[0]       #index of array
num_px = train_set_x_orig.shape[1]      #64 is at index 1 of array
num_hidden=50                           #number of hidden neurons

print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))

# Reshape the training and test examples

### START CODE HERE ### (≈ 2 lines of code)
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
### END CODE HERE ###

print ("\ntrain_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))

#standardize
train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255

# sigmoid

def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """

    ### START CODE HERE ### (≈ 1 line of code)
    s = 1/(1+np.exp(-z))
    ### END CODE HERE ###
    
    return s

#print ("sigmoid([0, 2]) = " + str(sigmoid(np.array([0,2]))))

def relu(z):
    """
    compute relu of z
    """
    s = np.maximum(0,z)
    return s

def reluDerivative(x):
     x[x<=0] = 0
     x[x>0] = 1
     return x

# GRADED FUNCTION: initialize_with_zeros

def initialize_with_zeros(hidden,dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    
    ### START CODE HERE ### (≈ 1 line of code)
    w1 = np.random.randn(hidden,dim)*0.01      #w should be random small value so * with 0.01      
    b1 = np.zeros((hidden,1))                   #shape of b = (hidden,1)
    w2 = np.random.randn(1,hidden)*0.01
    b2 = 0
    ### END CODE HERE ###

    assert(w1.shape == (hidden,dim))
    assert(b1.shape == (hidden,1))
    assert(w2.shape == (1,hidden))
    assert(isinstance(b2, float) or isinstance(b2, int))
    
    return w1, b1, w2, b2

dim = num_px*num_px*3        #set dimension of w=(num_px × num_px × 3, 1)
w1, b1, w2, b2 = initialize_with_zeros(num_hidden,dim)
print ("w1 = " ,w1.shape)     
print ("b1 = " ,b1.shape)
print ("w2 = " ,w2.shape)     
print ("b2 = " ,b2)

# GRADED FUNCTION: propagate

def propagate(w1, b1, w2, b2, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    
    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """
    
   # print("w2",w2.shape)
    
    m = X.shape[1]
    # FORWARD PROPAGATION (FROM X TO COST)
    ### START CODE HERE ### (≈ 2 lines of code)
    Z1 = (np.dot(w1,X)+b1)
    A1 = relu(Z1)       #relu on hidden layer
    #print("A1",A1.shape)
    Z2 = np.dot(w2,A1)+b2
    #print("Z2",Z2.shape)
    A2 = sigmoid(Z2)                             # compute sigmoid activation on output layer
    
    cost = (- 1 / m) * np.sum(Y * np.log(A2) + (1 - Y) * (np.log(1 - A2)))                                # compute cost
    ### END CODE HERE ###
    
    # BACKWARD PROPAGATION (TO FIND GRAD)
    ### START CODE HERE ### (≈ 2 lines of code)
    #print("A2",A2.shape)
    #print("Y",Y.shape)
    dz2 = A2-Y
    #print("dz2",dz2.shape)
    dw2 = np.dot(dz2,np.transpose(A1))/m
    db2 = np.sum(dz2,axis=1,keepdims=True)/m
    #print("dw2",dw2.shape)
    #print("db2",db2.shape)
    
    dz1 = np.dot(np.transpose(w2),dz2)*reluDerivative(Z1)
    dw1 = np.dot(dz1,np.transpose(X))/m
    db1 = np.sum(dz1,axis=1,keepdims=True)/m
    ### END CODE HERE ###

   # assert(dw.shape == w.shape)
   # assert(db.dtype == float)
    cost = np.squeeze(cost)
   # assert(cost.shape == ())
    
    grads = {"dw2": dw2,
             "db2": db2,
             "dw1": dw1,
             "db1": db1}
    
    return grads, cost

#w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])
#grads, cost = propagate(w, b, X, Y)
#print ("dw = " + str(grads["dw"]))
#print ("db = " + str(grads["db"]))
#print ("cost = " + str(cost))

# GRADED FUNCTION: optimize

def optimize(w1, b1, w2, b2, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    
    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """
    
    costs = []
    
    for i in range(num_iterations):
        
        
        # Cost and gradient calculation (≈ 1-4 lines of code)
        ### START CODE HERE ### 
        grads, cost = propagate(w1, b1, w2, b2, X, Y)
        ### END CODE HERE ###
        
        # Retrieve derivatives from grads
        dw1 = grads["dw1"]
        db1 = grads["db1"]
        dw2 = grads["dw2"]
        db2 = grads["db2"]
        
        # update rule (≈ 2 lines of code)
        ### START CODE HERE ###
        w1 = w1-np.multiply(learning_rate,dw1)
        b1 = b1-np.multiply(learning_rate,db1)
        w2 = w2-np.multiply(learning_rate,dw2)
        b2 = b2-np.multiply(learning_rate,db2)
        ### END CODE HERE ###
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w1": w1,
              "b1": b1,
              "w2": w2,
              "b2": b2}
    
    grads = {"dw1": dw1,
             "db1": db1,
             "dw2": dw2,
             "db2": db2}
    
    return params, grads, costs

#params, grads, costs = optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)
#
#print ("w = " + str(params["w"]))
#print ("b = " + str(params["b"]))
#print ("dw = " + str(grads["dw"]))
#print ("db = " + str(grads["db"]))

# GRADED FUNCTION: predict

def predict(w1,b1,w2,b2,X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w1 = w1.reshape(num_hidden,X.shape[0])
    
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    ### START CODE HERE ### (≈ 1 line of code)
    A1 = relu(np.dot(w1,X)+b1)
    A2 = sigmoid(np.dot(w2,A1)+b2)
    ### END CODE HERE ###
    
    for i in range(A2.shape[1]):
        
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        ### START CODE HERE ### (≈ 4 lines of code)
        if A2[0,i]<=0.5:
            Y_prediction[0,i]=0
        else: 
            Y_prediction[0,i]=1
        ### END CODE HERE ###
    
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction

#w = np.array([[0.1124579],[0.23106775]])
#b = -0.3
#X = np.array([[1.,-1.1,-3.2],[1.2,2.,0.1]])
#print ("predictions = " + str(predict(w, b, X)))

# GRADED FUNCTION: model

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    """
    Builds the logistic regression model by calling the function you've implemented previously
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """
    
    ### START CODE HERE ###
    
    # initialize parameters with zeros (≈ 1 line of code)
    w1, b1, w2, b2 = initialize_with_zeros(num_hidden, num_px*num_px*3)

    # Gradient descent (≈ 1 line of code)
    parameters, grads, costs = optimize(w1,b1,w2,b2,X_train,Y_train,num_iterations, learning_rate, print_cost)
    
    # Retrieve parameters w and b from dictionary "parameters"
    w1 = parameters["w1"]
    b1 = parameters["b1"]
    w2 = parameters["w2"]
    b2 = parameters["b2"]
    
    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = predict(w1,b1,w2,b2,X_test)
    Y_prediction_train = predict(w1,b1,w2,b2,X_train)

    ### END CODE HERE ###

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w1" : w1, 
         "b1" : b1,
         "w2" : w2,
         "b2" : b2,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

# Plot learning curve (with costs)
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()