import torch
import torch.nn as nn
from torchinfo import summary


# Version 3
class Model(nn.Module):

    def __init__(self, num_features):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(num_features, 3),  # index 0
            nn.ReLU(),                   # index 1
            nn.Linear(3, 1),             # index 2
            nn.Sigmoid()                 # index 3
        )

    def forward(self, features):
        out = self.network(features)
        return out


# Create input data
feature = torch.rand(10, 5)

# Create model
# feature.shape[1] = 5 because we have 5 features
model = Model(feature.shape[1])

# Calling forward function
print(model(feature))

# Show weights and bias 
print(model.network[0].weight) #weights and bias of first layer
print(model.network[0].bias)   #weights and bias of first layer

print('\n')

print(model.network[2].weight) #weights and bias of second layer
print(model.network[2].bias)   #weights and bias of second layer



# Visualize model
print(summary(model, input_size=(10, 5)))

"""
How the Sequential model works

Your network is:

Input
  ↓
Linear(5 → 3)
  ↓
ReLU
  ↓
Linear(3 → 1)
  ↓
Sigmoid
  ↓
Output

network[0] → Linear(5, 3)
network[1] → ReLU
network[2] → Linear(3, 1)
network[3] → Sigmoid


"""