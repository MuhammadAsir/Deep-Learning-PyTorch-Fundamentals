"""
# Batch Normalization (BatchNorm) Workflow in Deep Learning

Batch Normalization is a technique used in deep learning to normalize the activations of 
a neural network layer using the mean and variance of the current mini-batch. 
After normalization, it applies learnable scale (γ) and shift (β) parameters. 
Its main purpose is to keep activations at a more stable scale, 
which can make training faster and more stable

Covariate shift occurs when the distribution of input features changes between training and testing,
while the conditional relationship between the input and target remains approximately the same.

Batch Normalization was introduced to address the problem of internal covariate shift, 
where the distribution of activations changes as the parameters of previous layers 
are updated during training.

Batch Normalization normalizes the activations of a neural network
using statistics from the current mini-batch.

Workflow:

Input
  ↓
Linear / Dense Layer
  ↓
Batch Normalization
  ↓
Activation Function (e.g., ReLU)
  ↓
Next Layer


# Step 1: Linear Layer

The Linear layer produces activations.

Example:

[2, 4, 6]


# Step 2: Calculate Batch Mean

BatchNorm calculates the mean of the current mini-batch.

Mean = (2 + 4 + 6) / 3
     = 4


# Step 3: Calculate Batch Standard Deviation

BatchNorm calculates the standard deviation of the current
mini-batch.

Std ≈ 1.63


# Step 4: Normalize

Formula:

x̂ = (x - mean) / sqrt(variance + ε)

Ignoring ε for simplicity:

2 → (2 - 4) / 1.63 = -1.23
4 → (4 - 4) / 1.63 =  0
6 → (6 - 4) / 1.63 = +1.23

Normalized values:

[-1.23, 0, 1.23]

Now:

Mean ≈ 0
Std  ≈ 1


# Step 5: Scale and Shift

BatchNorm has two learnable parameters:

γ (gamma) = scale
β (beta)  = shift

Formula:

y = γx̂ + β

Initially:

γ = 1
β = 0

But during training, γ and β are learned.


# Step 6: Activation Function

The BatchNorm output is passed to the activation function.

Example:

BatchNorm output:
[-1.23, 0, 1.23]

ReLU:

[0, 0, 1.23]


# Complete Workflow

Input
  ↓
Linear Layer
  ↓
Calculate Batch Mean & Variance
  ↓
Normalize
  ↓
Scale (γ)
  ↓
Shift (β)
  ↓
Activation Function (ReLU)
  ↓
Next Layer


# Training vs Testing

During Training:

Current mini-batch
       ↓
Calculate mean & variance
       ↓
Normalize
       ↓
Apply γ and β
       ↓
Update model parameters
       ↓
Update running mean & variance


During Testing / Inference:

New input
    ↓
Use running mean & variance
    ↓
Normalize
    ↓
Apply γ and β
    ↓
Next layer


# Key Point

BatchNorm does NOT assume that the input has:

mean = 0
std  = 1

Instead, it calculates the mean and standard deviation
from the current mini-batch and then normalizes the activations
so that they have approximately:

mean ≈ 0
std  ≈ 1

Then γ and β allow the network to learn the appropriate
scale and shift.


"""