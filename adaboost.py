from time import time
from functools import reduce
from math import exp
import numpy as np
import cv2 as cv
from tensorflow.keras import datasets

[(x_train,y_train),(x_test,y_test)]=datasets.mnist.load_data()
cv.imshow('Yen Nhi',x_train[0])
cv.waitKey()

def DenseNet():
    layers=[]
    feed_forward_output=[]

    def layer(num_inputs,num_outputs):
        nonlocal layers
        layers+=[np.random.rand(num_outputs,num_inputs+1)]

    def feed_forward(input_layer):
        nonlocal layers,feed_forward_output
        feed_forward_output=[None for layer in layers]
        current_input=input_layer
        for idx in range(len(layers)):
            current_input=np.concatenate((current_input,[1]))
            feed_forward_output[idx]=1/(1+np.exp(-np.dot(layers[idx],current_input)))
            current_input=feed_forward_output[idx]
        return current_input

    def back_propagation(true_label):
        nonlocal layers,feed_forward_output
        layers[-1]+=np.dot(true_label*feed_forward_output[-1]*(1-feed_forward_output[-1]).reshape(-1,1),feed_forward_output[-2].reshape(1,-1))
        for idx in range(len(layers),0,-1):
            x=feed_forward_output[idx-1]
            delta_x

    def backPropagationHiddenLayer(_output,layer,delta):
        return _output

    layer=Layer(10,10)
    _input=[-5,-4,-3,-2,-1,0,1,2,3,4]
    print(feedForward(_input,layer))