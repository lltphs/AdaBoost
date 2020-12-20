from time import time
from functools import reduce
from math import exp
import numpy as np
import cv2 as cv
from tensorflow.keras import datasets,layers,Model
from ast import literal_eval

[(x_train,y_train),(x_test,y_test)]=datasets.mnist.load_data()
# cv.imshow('Yen Nhi',x_train[0])
# cv.waitKey()
print(x_train.shape)
print(y_train.shape)
class DenseNet:
    def __init__(self,num_inputs,lr=0.1):
        self.num_inputs=num_inputs
        self.lr=lr
        self.W=[]
        self.o=[]
        self.dL_over_dW=[]

    def layer(self,num_outputs):
        if len(self.W)==0:
            self.W+=[np.random.rand(num_outputs,self.num_inputs+1)]
        else:
            self.W+=[np.random.rand(num_outputs,self.W[-1].shape[0]+1)]

    def fit(self,x,y):
        self.x=x
        self.y=y
        for i in range(x.shape[0]):
            self.feed_forward(i)
            self.back_propagation(i)
        self.save()

    def feed_forward(self,i):
        self.o=[*[self.x[i]],*[None for layer in self.W]]
        for idx in range(len(self.W)):
            self.o[idx+1]=1/(1+np.exp(-np.dot(self.W[idx],np.concatenate((self.o[idx],[[1]])))))

    def back_propagation(self,i):
        self.dL_over_dW=[None for layer in self.W]
        dL_over_do=-(self.y[i]-self.o[-1])
        for idx in range(len(self.W),0,-1):
            self.dL_over_dW[idx-1]=np.dot(dL_over_do*self.o[idx]*(1-self.o[idx]),np.concatenate((self.o[idx-1],[[1]])).T)
            dL_over_do=np.dot(self.W[idx-1][:,:-1].T,dL_over_do*self.o[idx]*(1-self.o[idx]))
            self.W[idx-1]-=self.dL_over_dW[idx-1]
    
    def predict(self,x):
        return reduce(lambda input_layer,w:1/(1+np.exp(-np.dot(w,np.concatenate((input_layer,[[1]]))))),self.W,x)

    def save(self):
        file=open('weights','w')
        file.write(str(self.W))
        file.close()
    
X=np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]).reshape(36,1,1)
y=np.array([[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0]]).reshape(36,3,1)
X=x_train.reshape(60000,28*28,1)
Y=np.array([[i==y_train for i in range(9)] for j in range(y_train.shape[0])]).reshape(60000,10)
densenet=DenseNet(10)
densenet.layer(3)
densenet.layer(3)
densenet.fit(X,y)
print(densenet.predict(X[0]))
print(densenet.predict(X[31]))

inp=layers.Input(shape=(X.shape[1],X.shape[2]))
out=layers.Dense(10,activation='sigmoid')(inp)
model=Model(inputs=inp,outputs=out)
model.compile(optimizer='SGD',loss='MSE',metrics=['accuracy'])
model.fit(X,y,epochs=100)
print(model.predict(np.array([X[0],X[31]])))