import torch

"""
Why we need to clear grad?
In simple, when we perform backpropagation in PyTorch,
the gradients are accumulated in the .grad attribute of the tensors.

Calculate gradient
      ↓
Update weights
      ↓
CLEAR gradients
      ↓
Calculate new gradient
      ↓
Update weights
      ↓
CLEAR gradients

"""
import torch


x = torch.tensor(2.0, requires_grad=True)



# FIRST BACKWARD PASS

y = x**2

# dy/dx = 2(2) = 4

y.backward()

print("After first backward:")
print("x.grad =", x.grad)


# SECOND BACKWARD PASS

y = x**2

y.backward()

print("\nAfter second backward WITHOUT clearing:")
print("x.grad =", x.grad)

# Why is it 8 instead of 4?
#
# First backward: gradient = 4
# Second backward: gradient = 4
# PyTorch ACCUMULATES gradients: 4 + 4 = 8
# PyTorch does NOT automatically replace the old gradient.


# CLEAR THE GRADIENT
# Remove the old gradient
x.grad.zero_()

print("\nAfter clearing gradient:")
print("x.grad =", x.grad)



# THIRD BACKWARD PASS

y = x**2

y.backward()

print("\nAfter third backward:")
print("x.grad =", x.grad)


# Now the gradient is 4 again because: We cleared the old gradient first.
#Old gradient = 0 , New gradient = 4 , Total = 0 + 4 = 4


