"""
What is torch.nn? It is a core libary for building neural networks in PyTorch. 
It provides a set of modules and classes that help in creating and training deep learning models
efficiently and effectively. 
The torch.nn module includes various layers, loss functions, activation functions, and 
utilities that make it easier to define and train neural networks.

Key Components of torch.nn: 
1. Modules: The building blocks of neural networks. 
             Each module represents a layer or a component of the network, such as Linear,
             Conv2d, LSTM, etc.

2. Loss Functions: Predefined loss functions like CrossEntropyLoss, MSELoss, etc., 
                   that are used to compute the error between predicted and actual values.

3. Activation Functions: Functions like ReLU, Sigmoid, Tanh, etc., 
                         that introduce non-linearity into the model.

4.Containers: Classes like Sequential and ModuleList that help in organizing and 
              managing multiple modules.It allows us to stack layers and create complex architectures easily.

5.Regularization and Dropout: Techniques to prevent overfitting, such as Dropout layers and 
                              weight decay.

Advantages of using torch.nn:
Easy to build neural networks — provides ready-made components.
Pre-built layers — such as Linear, Conv2d, and LSTM.
Pre-built loss functions — such as MSELoss and CrossEntropyLoss.
Automatic parameter management — handles weights and biases for us.
Less code — we don't need to implement common neural-network operations from scratch.


When torch.nn  needed most?
torch.nn is needed when we are building a neural network in PyTorch. 
It gives us ready-made tools like layers, activation functions, and 
loss functions, so we don't have to implement them from scratch

The main disadvantage is that it can hide the underlying mathematical operations and 
give us less low-level control.Because many things are already implemented for you, 
you may have less control over the low-level details.

"""