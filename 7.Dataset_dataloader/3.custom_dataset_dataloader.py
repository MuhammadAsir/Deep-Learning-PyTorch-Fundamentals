from sklearn.datasets import make_classification
from torch.utils.data import Dataset,DataLoader
import torch

X,y=make_classification(
    n_samples=10,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_classes=2,
    random_state=42
)
print(X)


class custom_Dataset(Dataset):
      def __init__(self,features,labels):

            self.features=features
            self.labels=labels
      
      def __len__(self):
            return self.features.shape[0]

      def __getitem__(self,index) :
            return self.features[index],self.labels[index]


dataset=custom_Dataset(X,y)

print(len(dataset))
print(dataset[2])


dataloader=DataLoader(dataset,batch_size=2,shuffle=True)

for batch_feature,batch_labels in dataloader:
      print(batch_feature)
      print(batch_labels)

      




"""
# Why Do We Need Custom Dataset in PyTorch?

We don't always need a Custom Dataset.

If our data is already in tensors like:

X → input features
y → target

we can simply use:


from torch.utils.data import TensorDataset

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)


But sometimes our data has a custom structure, for example:

* Images stored in folders
* CSV files
* Text files
* Different preprocessing requirements
* Need to load and transform each sample differently

In these cases, we create a **Custom Dataset**.

## What problem does Custom Dataset solve?

It tells PyTorch:

1. How many samples are in my dataset
2. How to get one particular sample
3. How to process that sample before giving it to the model

We create a Custom Dataset by inheriting from `Dataset`:


from torch.utils.data import Dataset

class MyDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


### `__len__()`

Tells PyTorch how many samples are available.


def __len__(self):
    return len(self.X)


### `__getitem__()`

Tells PyTorch how to get one sample.


def __getitem__(self, index):
    return self.X[index], self.y[index]


For example:


index = 0

X[0] → input
y[0] → target

return X[0], y[0]


## Why not always use TensorDataset?

`TensorDataset` is already prepared for a simple situation:


X → tensors
y → tensors


It automatically keeps:


X[0] ↔ y[0]
X[1] ↔ y[1]
X[2] ↔ y[2]


So if our data is simple, `TensorDataset` is enough.

If we need special logic for loading or processing the data, we use a **Custom Dataset**.

## Simple Difference


TensorDataset
    ↓
Simple data
X and y are already available as tensors

Custom Dataset
    ↓
Our own data structure
We decide how each sample is loaded and processed


## Then DataLoader

After creating the dataset, we can use:


from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)


`DataLoader` takes the dataset and gives the model data in batches.


Custom Dataset / TensorDataset
            ↓
        DataLoader
            ↓
      Batch of samples
            ↓
          Model


### Remember


Dataset
→ Defines how data is accessed

TensorDataset
→ Ready-made Dataset for tensors

Custom Dataset
→ We define our own way to load/process data

DataLoader
→ Gives dataset samples to the model in batches

"""