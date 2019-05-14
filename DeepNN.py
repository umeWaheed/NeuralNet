# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 17:22:28 2019

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

m_train = train_set_x_orig.shape[0]     #index of the array
m_test = test_set_x_orig.shape[0]       #index of array
num_px = train_set_x_orig.shape[1]      #64 is at index 1 of array
dim=num_px*num_px*3
num_hidden=[dim,1350,1200,150,1];       #number of hidden neurons
layers=4                                #number of layers



print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))

# Reshape the training and test examples

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

print ("\ntrain_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))

#standardize
train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255

#sigmoid of z
def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s

#derivative of sigmoid
def sigmoidDerivative(z):
    s = sigmoid(z);
    return s*(1-s)

#relu of z
def relu(z):
    s = np.maximum(0,z)
    return s

#derivative of relu 
def reluDerivative(x):
     x[x<=0] = 0
     x[x>0] = 1
     return x

#initialize_with_zeros
def initialize_with_zeros():
    W = []
    B = []
    W.append(np.random.randn(1,1)*0.01)     #dummy weights for layer 0
    B.append(np.zeros((1,1)))     			#dummy bais for layer 0
    
    for i in range(1,layers+1,1):
        curr = num_hidden[i]        # neurons in layer l
        prev = num_hidden[i-1]      # neurons in layer l-1
        W.append(np.random.randn(curr,prev)*0.01)           #w should be random small value so * with 0.01
        print(W[i].shape)
        B.append(np.zeros((curr,1)))                         #shape of b = (hidden,1)
    return W,B

#forward propagation    
def propagate(W, B, X, Y):
    m = X.shape[1]
    
    Z = []      		#cache z's
    A = []      		#cache activations
    A.append(X)     	#a[0] is X
    Z.append(np.zeros((1,1)))   #dummy
    
    for i in range(1,layers,1):         #iterate till the 2nd last layer for relu
        z=np.dot(W[i],A[i-1])+B[i]
        Z.append(z)     	            #store z_i
        A.append(relu(z))        		#store activation_i relu on hidden layers
           
    z = np.dot(W[layers],A[layers-1])+B[layers]
    Z.append(z)
    A.append(sigmoid(z))        #sigmoid on output layer 
    
    cost = (- 1 / m) * np.sum(Y * np.log(A[layers]) + (1 - Y) * (np.log(1 - A[layers])))  # compute cost
    
    # BACKWARD PROPAGATION (TO FIND GRAD)
    
    dZ = [None]*(layers+1)     #empty lists with size layers+1
    dA = [None]*(layers+1)
    dW = [None]*(layers+1)
    dB = [None]*(layers+1)
    
    dA[layers]= -(np.divide(Y, A[layers]) - np.divide(1 - Y, 1 - A[layers]))
    
    #find sigmoid derivative for the last layer
    dZ[layers] = np.multiply(dA[layers],sigmoidDerivative(Z[layers])) 
    dW[layers] = np.dot(dZ[layers],np.transpose(A[layers-1]))/m
    dB[layers] = np.sum(dZ[layers],axis=1,keepdims=True)/m
    dA[layers-1] = np.dot(np.transpose(W[layers]),dZ[layers])
    
    for i in range(layers-1,0,-1):         #iterate from last till the 1st hidden layer for relu
        dZ[i] = dA[i]*reluDerivative(Z[i])
        dW[i] = np.dot(dZ[i],np.transpose(A[i-1]))/m
        dB[i] = np.sum(dZ[i],axis=1,keepdims=True)/m
        dA[i-1] = np.dot(np.transpose(W[i]),dZ[i])
        
    cost = np.squeeze(cost)
    assert(cost.shape == ())
   
    grads = {"dW": dW,
             "dB": dB
             }

    return grads, cost
    
#optimize
def optimize(W, B, X, Y, num_iterations, learning_rate, print_cost = False):    
    costs = []
    
    for j in range(num_iterations):
		grads, cost = propagate(W, B, X, Y)
        
        # Retrieve derivatives from grads
        dW = grads["dW"]
        dB = grads["dB"]
        
		for i in range(1,layers+1,1):
            W[i] = W[i]-np.multiply(learning_rate,dW[i])
            B[i] = B[i]-np.multiply(learning_rate,dB[i])
        
        # Record the costs
        if j % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training examples
        if print_cost and j % 100 == 0:
            print ("Cost after iteration %i: %f" %(j, cost))
        
    
    params = {"W": W,
              "B": B}
    
    grads = {"dW": dW,
             "dB": dB}
    
    return params, grads, costs


#predict for both training and test samples
def predict(W, B, X):
    m = X.shape[1]

    Y_prediction = np.zeros((1,m))
    W[1] = W[1].reshape(num_hidden[1],X.shape[0])   #1st hidden, input features

    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A1=[]
    A1.append(X) #activation for layer 0
        
    for i in range(1, layers, 1):
        A1.append(relu(np.dot(W[i],A1[i-1])+B[i]))
   
    A2 = sigmoid(np.dot(W[layers],A1[layers-1])+B[layers])
     
    for i in range(A2.shape[1]):
        
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if A2[0,i]<=0.5:
            Y_prediction[0,i]=0
        else: 
            Y_prediction[0,i]=1
    
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    # initialize parameters with zeros 
    W, B = initialize_with_zeros()

    # Gradient descent 
    parameters, grads, costs = optimize(W,B,X_train,Y_train,num_iterations, learning_rate, print_cost)
    
    # Retrieve parameters w and b from dictionary "parameters"
    W = parameters["W"]
    B = parameters["B"]
    
    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = predict(W, B, X_test)
    Y_prediction_train = predict(W, B, X_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "W" : W, 
         "B" : B,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = 0.07, print_cost = True)

# Plot learning curve (with costs)
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()
