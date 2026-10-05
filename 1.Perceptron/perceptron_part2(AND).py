
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self,L_rate,epochs):
        self.w=np.random.randn(3) * 1e-4#Initialize the weights of the perceptron with small random values. The weights are initialized to a 1D array of size 3 (for two input features and one bias term) using a standard normal distribution, scaled by 1e-4 to keep the initial weights small.
        print("Initial weights:",self.w)
        self.L_rate=L_rate
        self.epochs=epochs

    def activation_function(self,input,w):
        z=np.dot(input,w)
        return np.where(z>0,1,0)

    def fit(self,X,y):
        self.X=X
        self.y=y
        x_with_bias=np.c_[self.X,-np.ones((len(self.X),1))]#Concatenate the bias term with the input features X along the second axis (columns). This adds a bias term to each sample in the dataset.
        
        for epoch in range(self.epochs):
         y_hat=self.activation_function(x_with_bias,self.w)
         print(f"Predicted output after epoch {epoch+1}:",y_hat)
         error=self.y-y_hat 
         print(f"Error after epoch {epoch+1}:",error)
         #weight update
         self.w=self.w+self.L_rate*np.dot(x_with_bias.T,error) #we have transposed the input data (x_with_bias.T) to match the dimensions for matrix multiplication with the error vector. This allows us to compute the weight updates for each feature and the bias term simultaneously.

    def predict(self,X):
        x_with_bias=  np.c_[X,-np.ones((len(X),1))]
        return self.activation_function(x_with_bias,self.w)
    
#the predict function is used to make predictions on new input data after the perceptron has been trained. It takes the input features X, adds a bias term, and applies the activation function to compute the predicted output based on the learned weights.



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

model=Perceptron(L_rate=0.1,epochs=10)
model.fit(X,Y)

y_pred=model.predict(X)
print("Predicted output after training:",y_pred)
"""
1. The Output: "Hard" vs. "Soft"
Perceptron (Your code): Uses a Step Function. It only knows two things: 0 or 1.There is no middle ground. It's like a light switch.

Logistic Regression: Uses a Sigmoid Function. It outputs a probability (like 0.85 or 0.12). It says, "I am 85% sure this is a 1." It’s like a dimmer switch.


2.The Activation Function
Look at how the math changes:
Perceptron: return np.where(z > 0, 1, 0)
Logistic Regression: return 1 / (1 + np.exp(-z))


"""