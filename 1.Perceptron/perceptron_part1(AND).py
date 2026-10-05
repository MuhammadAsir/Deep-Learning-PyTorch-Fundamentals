"""
Perceptron is a simple linear binary classifier.
 It is a type of artificial neural network that consists of a single layer of neurons. 
 The perceptron algorithm is used for supervised learning, 
 where the model learns to classify input data into one of two classes based on 
 labeled training data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

w=np.array([2,2])

x=np.array([2,2])

print(np.dot(w,x))

random=np.random.randn(3)#It means generating 3 random numbers from a standard normal distribution (mean=0, std=1).
print(random)

data={
   "x1":[0,0,1,1],
    "x2":[0,1,0,1],
    "y":[0,0,0,1]
}
AND=pd.DataFrame(data)
print(AND)

X=AND.drop("y",axis=1)
Y=AND["y"]
Y.to_frame()#Convert the Series to a DataFrame
bias=-np.ones((len(X),1))#Create a bias term with -1 for each sample in the dataset and writing 1 as the second argument to specify that we want a column vector.


bias=np.c_[X,bias]#Concatenate the bias term with the input features X along the second axis (columns). This adds a bias term to each sample in the dataset.

print(bias)


