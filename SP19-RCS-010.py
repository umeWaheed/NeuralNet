#shallow nn
#SP19-RCS-010
#ASSIGNMENT-2
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
import math

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# number of training test samples and dimension of an image
m_train = train_set_x_orig.shape[0]     #index of the array
m_test = test_set_x_orig.shape[0]       #index of array
num_px = train_set_x_orig.shape[1]      #64 is at index 1 of array
num_hidden=10                           #number of hidden neurons

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

# standardize
train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255

# sigmoid
def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s

# relu
def relu(z):
    s = np.maximum(0,z)
    return s

# derivative of relu
def reluDerivative(x):
     x[x<=0] = 0
     x[x>0] = 1
     return x

# initialize_with_zeros
def initialize_with_zeros(hidden,dim):
    w1 = np.random.randn(hidden,dim)*0.01      # w should be random small value so * with 0.01      
    b1 = np.zeros((hidden,1))                   # shape of b = (hidden,1)
    w2 = np.random.randn(1,hidden)*0.01
    b2 = 0

    assert(w1.shape == (hidden,dim))
    assert(b1.shape == (hidden,1))
    assert(w2.shape == (1,hidden))
    assert(isinstance(b2, float) or isinstance(b2, int))
    
    return w1, b1, w2, b2

# FORWARD PROPAGATION (FROM X TO COST)
def propagate(w1, b1, w2, b2, X, Y):
    m = X.shape[1]
    Z1 = (np.dot(w1,X)+b1)
    A1 = relu(Z1)      				 # relu on hidden layer
    #print("A1",A1.shape)
    Z2 = np.dot(w2,A1)+b2
    A2 = sigmoid(Z2)                             # compute sigmoid activation on output layer
    
    cost = -(np.sum(np.multiply(Y,np.log(A2)))+np.sum(np.multiply(1-Y,np.log(1-A2))))/m                                 # compute cost
   
    # BACKWARD PROPAGATION (TO FIND GRAD)
   
    dz2 = A2-Y
    dw2 = np.dot(dz2,np.transpose(A1))/m
    db2 = np.sum(dz2,axis=1,keepdims=True)/m
    
    dz1 = np.dot(np.transpose(w2),dz2)*reluDerivative(Z1)
    dw1 = np.dot(dz1,np.transpose(X))/m
    db1 = np.sum(dz1,axis=1,keepdims=True)/m
   
    cost = np.squeeze(cost)
    
    grads = {"dw2": dw2,
             "db2": db2,
             "dw1": dw1,
             "db1": db1}
    
    return grads, cost

# GRADED FUNCTION: optimize
def optimize(w1, b1, w2, b2, X, Y, num_iterations, learning_rate, print_cost = False):    
    costs = []
    
    for i in range(num_iterations):    
        grads, cost = propagate(w1, b1, w2, b2, X, Y)
        
        # Retrieve derivatives from grads
        dw1 = grads["dw1"]
        db1 = grads["db1"]
        dw2 = grads["dw2"]
        db2 = grads["db2"]
        
        w1 = w1-np.multiply(learning_rate,dw1)
        b1 = b1-np.multiply(learning_rate,db1)
        w2 = w2-np.multiply(learning_rate,dw2)
        b2 = b2-np.multiply(learning_rate,db2)
        
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


# GRADED FUNCTION: predict
def predict(w1,b1,w2,b2,X):    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w1 = w1.reshape(num_hidden,X.shape[0])
    
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A1 = relu(np.dot(w1,X)+b1)
    A2 = sigmoid(np.dot(w2,A1)+b2)
    
    for i in range(A2.shape[1]):
        
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if A2[0,i]<=0.5:
            Y_prediction[0,i]=0
        else: 
            Y_prediction[0,i]=1
    
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction

# GRADED FUNCTION: model
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    # initialize parameters with zeros (≈ 1 line of code)
    w1, b1, w2, b2 = initialize_with_zeros(num_hidden, num_px*num_px*3)

    # Gradient descent (≈ 1 line of code)
    parameters, grads, costs = optimize(w1,b1,w2,b2,X_train,Y_train,num_iterations, learning_rate, print_cost)
    
    # Retrieve parameters w and b from dictionary "parameters"
    w1 = parameters["w1"]
    b1 = parameters["b1"]
    w2 = parameters["w2"]
    b2 = parameters["b2"]
    
    Y_prediction_test = predict(w1,b1,w2,b2,X_test)
    Y_prediction_train = predict(w1,b1,w2,b2,X_train)
    
    img = scipy.misc.imread("C:\\Users\\ACER\\Documents\\univ\\sem1\\nn\\SRCNN-Tensorflow-master\\Train\\t6.bmp", flatten=True)
    pre = predict(w1,b1,w2,b2,img);
    print(pre)

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

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = False)

# Plot learning curve (with costs)
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()