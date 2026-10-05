# Deep Learning & PyTorch Fundamentals

A structured collection of my Deep Learning fundamentals, mathematical concepts, and PyTorch implementations.

This repository focuses on understanding how neural networks learn, from the fundamentals of perceptrons and backpropagation to optimization, regularization, normalization, and complete neural network training pipelines using PyTorch.

## 📚 Topics Covered

### 1. Deep Learning Fundamentals

- Deep Learning fundamentals and real-world applications
- Machine Learning vs Deep Learning
- Neural Network types
- Artificial Neural Networks (ANN)
- Limitations of linear models
- Why neural networks are needed

### 2. Perceptron

- Perceptron concept and architecture
- Forward computation
- Perceptron training
- Loss functions
- Weight and bias updates
- Limitations of the Perceptron

### 3. Multi-Layer Perceptron (MLP)

- MLP architecture
- Input, hidden, and output layers
- Forward propagation
- Activation functions
- Building neural networks
- ANN-based classification and regression

### 4. Backpropagation

- Forward propagation
- Backpropagation concept and intuition
- Computational graph
- Chain rule
- Gradient calculation
- Weight and bias updates
- Mathematical understanding of backpropagation
- Backpropagation implementation with PyTorch

### 5. PyTorch Fundamentals

- PyTorch tensors
- Tensor operations
- Autograd
- Computational graphs
- `requires_grad`
- `backward()`
- Gradient calculation
- `nn.Module`
- `nn.Sequential`
- Building neural networks with PyTorch

### 6. PyTorch Training Pipeline

- Dataset preparation
- `TensorDataset`
- Custom Dataset
- `DataLoader`
- Batch training
- Mini-batch training
- Stochastic Gradient Descent
- Forward pass
- Loss calculation
- Backward pass
- Optimizer step
- Gradient resetting
- Model evaluation
- Complete training workflow

### 7. Gradient Descent & Optimization

- Batch Gradient Descent
- Mini-Batch Gradient Descent
- Stochastic Gradient Descent (SGD)
- Learning rate
- Convergence
- Oscillations during optimization
- SGD with Momentum
- AdaGrad
- RMSProp
- Adam

### 8. Loss Functions

#### Regression

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Huber Loss

#### Classification

- Binary Cross-Entropy (BCE)
- Categorical Cross-Entropy (CCE)

#### Concepts

- How loss functions guide learning
- Loss vs evaluation metrics
- Choosing an appropriate loss function

### 9. Training Challenges

- Slow convergence
- Oscillations
- Vanishing gradients
- Exploding gradients
- Poor weight initialization
- Learning-rate problems
- Feature scaling problems
- Early stopping
- Training stability

### 10. Feature Scaling & Standardization

- Why feature scaling matters in neural networks
- Standardization
- Normalization
- Effect of feature scale on gradients
- Scaled vs non-scaled training
- Impact of scaling on optimization

### 11. Regularization

- Overfitting
- L1 Regularization
- L2 Regularization
- Weight Decay
- Dropout
- How regularization reduces overfitting
- Relationship between L2 regularization and weight decay

### 12. Weight Initialization

- Importance of initialization
- Problems caused by poor initialization
- Weight initialization strategies
- Initialization and gradient flow
- Improving training stability

### 13. Batch Normalization

- What Batch Normalization does
- Batch mean and variance
- Normalizing activations
- Learnable scale and shift parameters
- Internal covariate shift concept
- Benefits of Batch Normalization
- Batch Normalization and training stability

## 🛠️ Technologies & Tools

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook
- Google Colab
- VS Code

## 📂 Repository Structure

```text
deep-learning-pytorch-fundamentals/
│
├── 01-Deep-Learning-Fundamentals/
├── 02-Perceptron/
├── 03-Backpropagation/
├── 04-Autograd/
├── 05-PyTorch-Training-Pipeline/
├── 06-PyTorch-nn-Module/
├── 07-Batch-MiniBatch-SGD/
├── 08-Dataset-DataLoader/
├── 09-Loss-Functions/
├── 10-Feature-Scaling/
├── 11-Batch-Normalization/
├── 12-Optimizers/
├── 13-Regularization/
├── 14-Training-Challenges/
│
└── README.md
```

👨‍💻 Author
Muhammad Asir Hossain Chowdhury

BSc in Computer Science & Engineering

Focus: AI / Machine Learning / Deep Learning
