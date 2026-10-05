"""
# Mini-Batch Gradient Descent: Problem & Solution

## Problem with Manual Mini-Batching

We can manually divide the training data into batches:


for start_index in range(0, sample, batch_size):

    end_index = start_index + batch_size

    X_batch = X_train_tensor[start_index:end_index]
    y_batch = y_train_tensor[start_index:end_index]


This works, but it has some problems:

### 1. Manual batch management

We have to manually calculate:

* `start_index`
* `end_index`
* Data slicing

This makes the training code longer and harder to maintain.

### 2. No shuffling

If we use:


X_train_tensor[start_index:end_index]

the data is always processed in the same order.

For example:


Batch 1 → samples 1–32
Batch 2 → samples 33–64
Batch 3 → samples 65–96
...


If the dataset has an ordering pattern, this can negatively affect training.

### 3. Less scalable

For large datasets, manually handling batches is inconvenient.

In real-world projects, datasets can contain thousands, millions, or more samples. 
We want PyTorch to handle batch creation and data loading for us.

---

# Solution: DataLoader

PyTorch provides `DataLoader` to handle mini-batches automatically.


from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)


Now we don't need to manually calculate `start_index` and `end_index`.

Instead:


for X_batch, y_batch in train_loader:

    y_pred = model(X_batch)

    loss = loss_function(
        y_pred,
        y_batch.reshape(-1, 1)
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


`DataLoader` automatically gives us:


Dataset
   ↓
DataLoader
   ↓
Batch 1 → 32 samples
   ↓
Batch 2 → 32 samples
   ↓
Batch 3 → 32 samples
   ↓
...


And because:


shuffle=True


the training data is shuffled at the beginning of each epoch.

---

# Why DataLoader is More Scalable

`DataLoader` provides a standard way to:

* Create mini-batches automatically
* Shuffle training data
* Load data efficiently
* Work with large datasets
* Use multiple workers for data loading when needed

Therefore:

Manual slicing
     ↓
Works, but harder to maintain


DataLoader
     ↓
Automatic batching + shuffling + scalable data loading


## Important Note

`torch.optim.SGD` is the optimizer used to update the model parameters.

The type of Gradient Descent depends on how much data is used for each update:


Entire dataset → Batch Gradient Descent

Small batch → Mini-Batch Gradient Descent

One sample → Stochastic Gradient Descent

So this:


optimizer = torch.optim.SGD(...)


does NOT automatically mean that the training is Stochastic Gradient Descent.

In our case:


DataLoader(batch_size=32)


means we are using **Mini-Batch Gradient Descent**.


"""