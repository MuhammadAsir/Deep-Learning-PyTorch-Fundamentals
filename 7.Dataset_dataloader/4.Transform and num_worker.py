"""
Transform and num_workers in PyTorch
1. What is Transform?

A transform is used to modify or preprocess data before giving it to the model.

For example, an image may need to be:

Resized
Converted to a tensor
Normalized
Flipped or cropped

Example:

from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

This means:

Original Image
      ↓
Resize to 224 × 224
      ↓
Convert to PyTorch Tensor
      ↓
Model

We can apply the transform like this:

image = transform(image)

Why do we use Transform?

Raw data is often not in the format or scale that the model expects.

So:

Transform = Prepare/modify each sample before giving it to the model


2. What is num_workers?

num_workers is a parameter of DataLoader.

It determines how many separate worker processes are used to load and prepare data.

Example:

from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

Here:

batch_size = 32
→ Load 32 samples in one batch

shuffle = True
→ Shuffle the training data

num_workers = 4
→ Use 4 worker processes to load/prepare data
Why do we need num_workers?

Suppose we have a large image dataset.

Without multiple workers:

Main Process
     ↓
Load batch
     ↓
Model trains
     ↓
Load next batch
     ↓
Model trains

The model may have to wait while the next batch is being loaded.

With multiple workers:

Worker 1 ──┐
Worker 2 ──┤
Worker 3 ──┼──→ DataLoader → Model
Worker 4 ──┘

Multiple workers can load/prepare data while the model is training.

This can reduce the time the model spends waiting for data.

Important for Windows / VS Code

When learning PyTorch on Windows, start with:

num_workers=0

Example:

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

Once everything works, you can try:

num_workers=2

or:

num_workers=4

The best value depends on your CPU, dataset, storage, and training workload.

Easy way to remember
Transform
→ What should I do to each sample?

num_workers
→ How many workers should help load the samples?

DataLoader
→ How should I provide those samples to the model?

Example:

Dataset
   ↓
Transform
   ↓
DataLoader
   ↓
num_workers help load data
   ↓
Batch
   ↓
Model


Sampler in PyTorch:

A Sampler decides which samples should be selected from a dataset and in what order.

Think of a Sampler as a selection/order manager for the DataLoader.

Example

Suppose our dataset has 5 samples:

Index:    0    1    2    3    4
Sample:   A    B    C    D    E

Sampler can decide the order:

0 → 1 → 2 → 3 → 4

or:

3 → 0 → 4 → 1 → 2

The Sampler only provides the indices.

Sampler
   ↓
Selects indices
   ↓
DataLoader
   ↓
Gets actual samples
   ↓
Creates batches
   ↓
Model
Why do we need a Sampler?

Most of the time, we don't need to create a Sampler manually.

For example:

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

When shuffle=True, PyTorch automatically changes the order of samples for training.

However, sometimes we need special sampling behavior.

Example: Imbalanced Dataset

Suppose:

Class 0 → 900 samples
Class 1 → 100 samples

The dataset is imbalanced.

We may want to select Class 1 samples more frequently during training.

We can use:

WeightedRandomSampler

to control how frequently different samples are selected.

Sampler vs shuffle
shuffle=True
→ Randomly changes the order of samples

Sampler
→ Gives us more control over which samples are selected
  and how they are selected
Easy way to remember
Dataset
→ Stores/defines the data

Sampler
→ Decides WHICH samples and WHAT ORDER

DataLoader
→ Takes those samples and creates BATCHES

Model
→ Learns from those batches
One-line definition

Sampler = Controls the selection and order of samples
           given to the DataLoader.

"""