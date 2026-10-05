import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('https://raw.githubusercontent.com/gscdit/Breast-Cancer-Detection/refs/heads/master/data.csv')
print(df.head())


df.drop(['id','Unnamed: 32'], axis=1, inplace=True)

X=df.drop(['diagnosis'],axis=1)
y=df['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(y.unique())

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

encoder=LabelEncoder()
y_train=encoder.fit_transform(y_train)
y_test=encoder.transform(y_test)

print("X_train :", X_train,'\n')
print("y_train :", y_train,'\n')


##Numpy arrays to PyTorch tensors

X_train_tensor = torch.from_numpy(X_train)
X_test_tensor = torch.from_numpy(X_test)
y_train_tensor = torch.from_numpy(y_train)
y_test_tensor = torch.from_numpy(y_test)


##Defining Model

class MySimpleNN():

  def __init__(self, X):
        self.weight=torch.rand(X.shape[1],1,dtype=torch.float64,requires_grad=True) 
        self.bias=torch.zeros(1,dtype=torch.float64,requires_grad=True)
   

  def forward(self, X):
        z=torch.matmul(X, self.weight) + self.bias
        y_pred=torch.sigmoid(z)
        return y_pred

  def loss_function(self, y_pred, y):
        epsilon=1e-7
        y_pred=torch.clamp(y_pred, epsilon, 1-epsilon)
        
        # Calculate loss
        loss = -(y_train_tensor * torch.log(y_pred) + (1 - y_train_tensor) * torch.log(1 - y_pred)).mean()
        return loss
    

learning_rate = 0.1
epochs = 25

model = MySimpleNN(X_train_tensor)

# define loop
for epoch in range(epochs):

  # forward pass
  y_pred = model.forward(X_train_tensor)

  # loss calculate
  loss = model.loss_function(y_pred, y_train_tensor)

  # backward pass
  loss.backward()

  # parameters update
  with torch.no_grad():
    model.weight -= learning_rate * model.weight.grad
    model.bias -= learning_rate * model.bias.grad

  # zero gradients
  model.weight.grad.zero_()
  model.bias.grad.zero_()

  # print loss in each epoch
  print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')\
  
  
print("Bias: ",model.bias,'\n')
print("Weight: ",model.weight,'\n')


# model evaluation

with torch.no_grad():
  y_pred = model.forward(X_test_tensor)
  y_pred = (y_pred > 0.9).float()
  accuracy = (y_pred == y_test_tensor).float().mean()
  print(f'Accuracy: {accuracy.item()}')
 

 #We use with no grad to avoid calculating gradients during evaluation, which saves memory and computation time.
  """
  Suppose, we have a model that has been trained on a dataset and 
  we want to evaluate its performance on a test set. 
  During evaluation, we don't need to compute gradients 
  because we are not updating the model's parameters. 
  By using with torch.no_grad(), we can disable gradient calculation, 
  which reduces memory usage and speeds up the evaluation process. 
  This is especially important when working with large models or datasets, 
  as it can significantly improve efficiency.

"""