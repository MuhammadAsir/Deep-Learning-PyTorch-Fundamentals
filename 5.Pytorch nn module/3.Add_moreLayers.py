import torch
import torch.nn as nn
from torchinfo import summary  

#We will create a more complex model with additional layers and activation functions. This will allow the model to learn more complex patterns in the data.

#Version 2

class Model(nn.Module):
      def __init__(self,num_features):
          super().__init__()    
          
          #layer 1
          self.linear1=nn.Linear(num_features,3) 
          self.relu=nn.ReLU()

          #layer 2
          self.linear2 = nn.Linear(3,1)
          self.sigmoid = nn.Sigmoid()


      def forward(self,features):
          out = self.linear1(features)
          out = self.relu(out)

          out = self.linear2(out)
          out = self.sigmoid(out)

          return out
      

feature=torch.rand(10,5)

#create model
model=Model(feature.shape[1]) #feature.shape[1] = 5, because we have 5 features in our dataset.

#calling forward function

print(model(feature))

#Show model weights and bias

print("Model weights of layer 1:",model.linear1.weight)
print("Model bias of layer 1:",model.linear1.bias)  
print('\n')
print("Model weights of layer 2:",model.linear2.weight)
print("Model bias of layer 2:",model.linear2.bias)

#Visualize (pip install torchinfo)

print(summary(model,input_size=(10,5))) 