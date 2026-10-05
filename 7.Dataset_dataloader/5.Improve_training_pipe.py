import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
from torchinfo import summary
from torch.utils.data import Dataset,DataLoader

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

#change dtype
X_train_tensor = torch.tensor(X_train_tensor, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_tensor, dtype=torch.float32)

y_train_tensor = torch.from_numpy(y_train)
y_test_tensor = torch.from_numpy(y_test)

y_train_tensor = torch.tensor(y_train_tensor,dtype= torch.float32)
y_test_tensor = torch.tensor(y_test_tensor, dtype=torch.float32)


##Defining Model

class MySimpleNN(nn.Module):

  def __init__(self, num_features):
        super().__init__()
        self.linear = nn.Linear(num_features,1)
        self.sigmoid = nn.Sigmoid()
   

  def forward(self,features):
        out = self.linear(features)
        out = self.sigmoid(out)
        return out
    
class custom_Dataset(Dataset):
      def __init__(self,features,labels):

            self.features=features
            self.labels=labels
      
      def __len__(self):
            return self.features.shape[0]

      def __getitem__(self,index) :
            return self.features[index],self.labels[index]

#Custom dataset
train_dataset=custom_Dataset(X_train_tensor,y_train_tensor)
test_dataset=custom_Dataset(X_test_tensor,y_test_tensor)

#DataLoader
train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=32,shuffle=False)

learning_rate = 0.1
epochs = 25

loss_function = nn.BCELoss() #It is used for binary classification problems, where the output is a probability value between 0 and 1. It measures the difference between the predicted probabilities and the actual binary labels (0 or 1) and calculates the loss accordingly.


model = MySimpleNN(X_train_tensor.shape[1])

#Optimization is the process of adjusting the parameters of a machine learning model to minimize the loss function. In this code, we are using Stochastic Gradient Descent (SGD) as the optimization algorithm. The optimizer updates the model's parameters based on the gradients computed during the backward pass, which helps the model learn and improve its predictions over time.
optimizer = torch.optim.SGD( model.parameters(), lr = learning_rate )


#Implementing Mini-Batch gradient
batch_size=32
epochs=25

#define loop

for epoch in range(epochs):

    for batch_feature,batch_labels in train_loader:
        
        #Forward pass
        y_pred=model(batch_feature)

        #Loss calculate
        loss = loss_function(y_pred,batch_labels.reshape(-1,1))
    
        #zero gradient
        optimizer.zero_grad()


        #backward pass
        loss.backward()


        #parameter update
        optimizer.step()

    #print loss in each epoch

    print(f" {epoch+1} , Loss = {loss.item()} ")




  


model.eval()
accuracy_list=[]

with torch.no_grad():
  for batch_feature,batch_labels in test_loader:
        y_pred = model(batch_feature)
        y_pred = (y_pred > 0.5).float()
        batch_accuracy = (y_pred.view(-1) == batch_labels).float().mean().item()
        accuracy_list.append(batch_accuracy)
        
print(f'Accuracy: {accuracy_list}')

overall_accuracy=sum(accuracy_list)/len(accuracy_list)
print(overall_accuracy)