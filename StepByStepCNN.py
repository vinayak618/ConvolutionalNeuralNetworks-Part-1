#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 14:29:17 2018

@author: vinayak
"""

"""
In this session, we will go through step by step analysis for CNN layers,
we will implement CONV and POOL layers using numpy library, 
we will implement forward prop for one layer and expand it to multiple layers.
"""

#import the packages required 
import numpy as np
import h5py
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
#plt.rcParams['image.interpolation'] = 'nearest'
#plt.rcParams['image.cmap'] = 'gray'


def zero_pad(X, pad):
    
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image.
    
    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions
    
    Returns:
    X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
    """
    
    #Here the padding is applied to the height and width of the image (m, n_H, n_W, n_C)
    X_pad = np.pad(X, ((0,0), (pad, pad),(pad,pad),(0,0)), 'constant', constant_values = 0)
    
    return X_pad
    

    
np.random.seed(1)
x = np.random.randn(4,3,3,2)
x_pad = zero_pad(x, 2)

print("Shape Before Padding : " , x.shape)
print("Shape After Padding : ", x_pad.shape)

#code for just plotting the data of X for visualization
fig, axarr = plt.subplots(1, 2)
axarr[0].set_title('x (Without Padding)')
axarr[0].imshow(x[0,:,:,0])
axarr[1].set_title('x_pad (With Padding(2))')
axarr[1].imshow(x_pad[0,:,:,0])

#Single Step of Convolution

"""
1. Takes the input
2. Applies filter to all the parts of input
3. outputs another volume of different size or maybe same size
"""

def conv_single_step(a_slice_prev, W, b):
    
    """
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation 
    of the previous layer.
    
    Arguments:
    a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
    W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)
    
    Returns:
    Z -- a scalar value, result of convolving the sliding window (W, b) on a slice x of the input data
    """
    
    #Elemnent-wise multiplication betn W and a_slice_prev(input)
    s = np.multiply(a_slice_prev, W)
    Z = np.sum(s)  #sum over entire s
    Z = Z = float(b)
    
    return Z

np.random.seed(1)
a_slice_prev = np.random.randn(4,4,3)
W = np.random.randn(4,4,3)
b = np.random.randn(1,1,1)

z = conv_single_step(a_slice_prev, W, b)
print("Output of Single CNN : ", z)

#CNN Forward pass

def conv_forward(A_prev, W, b, hparameters):
    
    """
    Implements the forward propagation for a convolution function
    
    Arguments:
    A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"
        
    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """
    
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape   #input shape
    (f,f,n_C_prev,n_C) = W.shape                       #filter shape
    
    #Extract the hyperparameters from the hparameters
    stride = hparameters["stride"]
    pad    = hparameters["pad"]
    
    #Output dimension, for more details of formula study about CNN basics(Deeplearning.ai Course4, week1)
    
    n_H = int((n_H_prev - f + 2 * pad)/stride) + 1
    n_W = int((n_H_prev - f + 2 * pad)/stride) + 1
    
    #initialize the output volume with zeros
    
    Z = np.zeros((m, n_H, n_W, n_C))
    
    A_prev_pad = zero_pad(A_prev, pad) #Padding for better computation
    
    for i in range(m):                                    #Loop over m training example
        a_prev_pad = A_prev_pad[i]                        #ith training Example
        for h in range(n_H):                              #Loop over Vertical axis of output volume
            for w in range(n_W):                          #Loop over Horizontal axis of output volume
                for c in range(n_C):                      #Loop over channel axis of output volume
                    #Finding the corner of current slice
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, W[...,c], b[...,c]) #values of W and b upto c(channels)
                    #We can even apply activation here
                    """ A[i, h, w, c] = activation (Z[i, h, w, c]) """
                    
                    
    assert(Z.shape == (m,n_H, n_W, n_C))                 #cross verifying the dimensions of Z
    cache = (A_prev, W, b, hparameters)
    
    return Z, cache                    
    
np.random.seed(1)   #just to make sure our and your values match together
A_prev = np.random.rand(10,4,4,3) #10 training example of size height=width = 4 and 3 channel input(RGB)
W = np.random.randn(2,2,3,8)  # 8 filters with size 2*2
b = np.random.randn(1,1,1,8)
hparameters = {"pad":2, "stride":2}
z, cache_conv = conv_forward(A_prev, W, b, hparameters)
print("Z's mean =", np.mean(z))
print("Z[3,2,1] =", z[3,2,1])
print("cache_conv[0][1][2][3] =", cache_conv[0][1][2][3])


#POOLING LAYER

"""
The pooling (POOL) layer reduces the height and width of the input. It helps reduce computation, 
as well as helps make feature detectors more invariant to its position in the input. 
The two types of pooling layers are:
Max-pooling layer: slides an ( f,f ) window over the input and stores the max value of the window in the output.
Average-pooling layer: slides an ( f,f) window over the input and stores the average value of the window in the output.
"""
def pool_forward(A_prev , hparameters, mode ="max"):
    
    """
    Implements the forward pass of the pooling layer
    
    Arguments:
    A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    hparameters -- python dictionary containing "f" and "stride"
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    
    Returns:
    A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters 
    """
    
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    f = hparameters["f"]
    stride = hparameters["stride"]
    
    #Output volume size
    n_H = int((n_H_prev - f)/stride) + 1
    n_W = int((n_H_prev - f)/stride) + 1
    n_C = n_C_prev
    
    #Initialize output volume with zeros
    
    A = np.zeros((m, n_H, n_W, n_C))
    
    ##Similar loop as of conv_forward for finding the corners of each training example
    
    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    
                    #find the corners
                    vert_start = h * stride
                    vert_end   = vert_start + f
                    horiz_start = w * stride 
                    horiz_end   = horiz_start + f
                    
                    a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]
                    
                    #Compute for the particular mode
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_prev_slice)
    
    cache = (A_prev, hparameters)
    
    assert(A.shape == (m, n_H, n_W, n_C))               #cross verifying the dimensions of A
    
    return A, cache
    

np.random.seed(1)
A_prev = np.random.randn(2, 4, 4, 3)
hparameters = {"stride" : 2, "f": 3}

A, cache = pool_forward(A_prev, hparameters)
print("mode = max")
print("A =", A)
print()
A, cache = pool_forward(A_prev, hparameters, mode = "average")
print("mode = average")
print("A =", A)

####           End of Forward propogation of CNN            ####


"""
Now coming back to the very important aspect of ML and Deep learning, that is Back prop. 
we will use some of the standard formulas which are available in notebook file, 
so please do check out the notebook file for more understanding regarding the same.

"""

def conv_backward(dZ, cache):               #cache has stored all the parameters.
    
    """
    Implement the backward propagation for a convolution function
    
    Arguments:
    dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward(), output of conv_forward()
    
    Returns:
    dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
               numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    dW -- gradient of the cost with respect to the weights of the conv layer (W)
          numpy array of shape (f, f, n_C_prev, n_C)
    db -- gradient of the cost with respect to the biases of the conv layer (b)
          numpy array of shape (1, 1, 1, n_C)
    """
    
    (A_prev, W, b, hparameters) = cache
    #retrive the dimensions of the A_prev(input) and W(filter)
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f,f,n_C_prev, n_C) = W.shape
    
    stride = hparameters["stride"]
    pad    = hparameters["pad"]
    
    #dimensions of dZ
    (m, n_H, n_W, n_C) = dZ.shape
    
    #Initialize the dA_prev, dW, db
    dA_prev = np.zeros((m , n_H_prev, n_W_prev, n_C_prev))
    dW      = np.zeros((f,f,n_C_prev,n_C))
    db      = np.zeros((1,1,1,n_C))
    
    #padding
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)
    
    #Almost same loop as conv_forward()
    
    for i in range(m):
        a_prev_pad  = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    
                    vert_start  = h * stride
                    vert_end    = vert_start + f
                    horiz_start = w * stride
                    horiz_end   = horiz_start + f
                    
                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]+= W[:,:,:,c] * dZ[i, h, w, c]
                    dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
                    db[:,:,:,c] += dZ[i, h, w, c]
         #Set the ith training example's dA_prev to the unpaded da_prev_pad (Hint: use X[pad:-pad, pad:-pad, :])
        dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
    
    assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev)) #Cross check the shape
    
    return dA_prev, dW, db

np.random.seed(1)
dA, dW, db = conv_backward(z, cache_conv) #z output of conv_forward()
print("dA_mean =", np.mean(dA))
print("dW_mean =", np.mean(dW))
print("db_mean =", np.mean(db))


#Backward propogation of POOLING LAYER

"""Here we need a helper function create_mask_from_window(), which helps us in keeping track of
max value in the matrix and its position, REFER (NOTEBOOK) for better intuition.
"""

def create_mask_from_window(x):
    
    """
    Creates a mask from an input matrix x, to identify the max entry of x.
    
    (1 for max value in matrix and 0 in non max value) i.e X = [1,2] -> M = [0,1]
    Arguments:
    x -- Array of shape (f, f)
    
    Returns:
    mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x.
    """
    mask = x == np.max(x) #Refer .ipynb file
    
    return mask
    
    
np.random.seed(1)
x = np.random.randn(2,3)
mask = create_mask_from_window(x)
print('x = ', x)
print("mask = ", mask)

#check the output, it will have maximum value corresponding to true and rest to false

""" Now helper function for average pooling , i.e dictributed_value() (Refer Notebook)"""

def distribute_value(dz, shape):
    
    """
    Distributes the input value in the matrix of dimension shape
    
    Arguments:
    dz -- input scalar
    shape -- the shape (n_H, n_W) of the output matrix for which we want to distribute the value of dz
    
    Returns:
    a -- Array of size (n_H, n_W) for which we distributed the value of dz
    """
    
    n_H, n_W = shape
    
    average = dz / (n_H * n_W)
    
    a = np.ones(shape) * average
    
    return a


a = distribute_value(2, (2,2))
print('distributed value =', a)


def pool_backward(dA, cache, mode = "max"):
    
    """
    Implements the backward pass of the pooling layer
    
    Arguments:
    dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
    cache -- cache output from the forward pass of the pooling layer, contains the layer's input and hparameters 
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    
    Returns:
    dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
    """
    
    (A_prev, hparameters) = cache
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    stride = hparameters["stride"]
    f    = hparameters["f"]
    
    (m, n_H, n_W, n_C) = dA.shape
    
    dA_prev = np.zeros(A_prev.shape)
    
    for i in range(m):
        a_prev = A_prev[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    
                    vert_start  = h * stride
                    vert_end    = vert_start + f
                    horiz_start = w * stride
                    horiz_end   = horiz_start + f
                    
                    if mode == "max":
                        # Use the corners and "c" to define the current slice from a_prev 
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        # Create the mask from a_prev_slice 
                        mask = create_mask_from_window(a_prev_slice)
                        # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA) 
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += np.multiply(mask, dA[i, h, w, c])
                        
                    elif mode == "average":
                        # Get the value a from dA 
                        da = dA[i, h, w, c]
                        # Define the shape of the filter as fxf 
                        shape = (f, f)
                        # Distribute it to get the correct slice of dA_prev. i.e. Add the distributed value of da. 
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += distribute_value(da, shape)
    
    assert(dA_prev.shape == A_prev.shape)           #Cross checking the shape
    
    return dA_prev

np.random.seed(1)
#A_prev = np.random.randn(2, 4, 4, 3)
#hparameters = {"stride" : 2, "f": 3}

A_prev = np.random.randn(10,5,5,3)
hparameters = {"stride" : 1, "f": 2}
A, cache = pool_forward(A_prev, hparameters)
dA = np.random.randn(5, 4, 2, 2)

dA_prev = pool_backward(dA, cache, mode = "max")
print("mode = max")
print('mean of dA = ', np.mean(dA))
print('dA_prev[1,1] = ', dA_prev[1,1])  
print()
dA_prev = pool_backward(dA, cache, mode = "average")
print("mode = average")
print('mean of dA = ', np.mean(dA))
print('dA_prev[1,1] = ', dA_prev[1,1]) 
                        
                    