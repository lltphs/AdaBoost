from time import time
from functools import reduce
from math import exp
from random import random

def DotProduct(X,W):
    return sum(map(lambda x:x[0]*x[1],zip(X+[1],W)))

def Sigmoid(realNumber):
    return 1/(1+exp(-realNumber))

def Layer(numOfInputs,numOfOutputs):
    return [[random() for noOfInp in range(numOfInputs)] for noOfOut in range(numOfOutputs)]

def feedForward(_input,layer):
    return [Sigmoid(DotProduct(_input,neural)) for neural in layer]

def backPropagationOutput(_input,_output,true_output,layer):
    return _output*(1-_output)(true_output-_output)*_input

def backPropagationHiddenLayer(_output,layer,delta):
    return _output

layer=Layer(10,10)
_input=[-5,-4,-3,-2,-1,0,1,2,3,4]
print(feedForward(_input,layer))