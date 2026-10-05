"""
# 1. What is an Optimizer?

An optimizer is used to update the model's weights and biases
so that the loss becomes smaller.

During training:

Input
  ↓
Model
  ↓
Prediction
  ↓
Loss
  ↓
Gradient
  ↓
Optimizer
  ↓
Updated weights and biases


The optimizer decides how the model's parameters (weights and biases)
should be changed using the gradients.

Example:

Before update:
weight = 0.5

After optimizer update:
weight = 0.43

The goal is to update the parameters so that the model makes
better predictions and the loss decreases.


--------------------------------------------------

# 2. Why do we need an Optimizer?

During backpropagation:

loss.backward()

PyTorch calculates the gradients of the model's parameters.

The gradient tells us:
"Which direction should the parameter move to reduce the loss?"

However, backward() only calculates the gradients.
It does NOT update the weights and biases.

The optimizer performs the actual update:

optimizer.step()


So:

loss.backward()
    ↓
Calculates gradients

optimizer.step()
    ↓
Updates weights and biases


A simple update formula is:

new weight = old weight - learning_rate × gradient


Example:

old weight = 2.0
gradient = 0.5
learning rate = 0.1

new weight = 2.0 - (0.1 × 0.5)
new weight = 1.95


Therefore:

backward() → calculates how the weight should change

optimizer.step() → actually changes the weight


3. How did training work WITHOUT an Optimizer?

Before using an optimizer, we could update the model's
weights and biases manually using the gradient.

The basic formula is:

new parameter = old parameter - learning_rate × gradient


Example:

weight = 2.0
gradient = 0.5
learning_rate = 0.1

new weight = 2.0 - (0.1 × 0.5)
new weight = 1.95


In PyTorch, we could do:

loss.backward()

with torch.no_grad():
    weight -= learning_rate * weight.grad


The problem is that a model can have many weights and biases.

For example:

W1
B1
W2
B2
W3
B3
...

We would have to update each parameter manually:

W1 -= lr * W1.grad
B1 -= lr * B1.grad
W2 -= lr * W2.grad
B2 -= lr * B2.grad


This becomes difficult and error-prone for large neural networks.


"""