"""
Autograd is a differentiation engine that powers neural network training in PyTorch. 
It provides automatic differentiation for all operations on Tensors.
It is needed for training neural networks because it allows us to compute gradientsa
automatically, 
which are essential for optimization algorithms like gradient descent.

"""
import torch

#Gradient = collection of derivatives with respect to multiple variables.

x=torch.tensor(1.0,requires_grad=True)
# requires_grad=True,tells PyTorch:Keep track of the mathematical operations involving x, because later I may want to calculate a gradient."

y=x**2
print(y) 
""" 
Output: tensor(1., grad_fn=<PowBackward0>),
The grad_fn is PyTorch saying:
"I remember how this value was calculated, so I can go backward later."

"""

y.backward()  # This computes the gradient of y with respect to x
print(x.grad)  # Output: tensor(2.), which is the derivative of y=x^2 with respect to x at x=1.0

#backward() tells PyTorch to perform the backward pass, which uses automatic differentiation to calculate gradients.

a=torch.tensor(2.0,requires_grad=True)
b=a**2

z=torch.sin(b)

z.backward()  # This computes the gradient of z with respect to a
print(a.grad)  # Output: tensor(-2.6146), which is the derivative of z=sin(a^2) with respect to a at a=2.0