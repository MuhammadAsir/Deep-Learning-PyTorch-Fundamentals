import torch
import torch.nn as nn
from torchinfo import summary   

"""
In this code,we will implement a simple neural network model using PyTorch's nn.Module class.

We use __init__() to initialize and define all the layers of our model. 
super().__init__() initializes the parent nn.Module, 
Allows PyTorch to properly recognize and manage the layers and their parameters.

class Model(nn.Module):
you say:
"I am making my own model, but I still need the parent's setup."
nn.Module = Parent
Model     = Child
"""

#Version 1
class Model(nn.Module):
      def __init__(self,num_features):
          super().__init__()    
          
          self.linear=nn.Linear(num_features,1) #Why 1? because we are predicting a single value (regression).
          self.Sigmoid=nn.Sigmoid() 

#Forward function is outside the init function remember, it is a separate function that defines how the input data flows through the model.
      def forward(self,features):
          out=self.linear(features)
          out=self.Sigmoid(out)
          return out
      

#create dataset
feature=torch.rand(10,5)

#create model
model=Model(feature.shape[1]) #feature.shape[1] = 5, because we have 5 features in our dataset.

#calling forward function

print(model(feature))

#Show model weights and bias

print("Model weights:",model.linear.weight)
print("Model bias:",model.linear.bias)  


#Visualize (pip install torchinfo)

print(summary(model,input_size=(10,5))) 