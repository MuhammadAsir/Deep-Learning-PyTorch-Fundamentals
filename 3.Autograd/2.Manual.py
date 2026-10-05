import torch


x=torch.tensor(2.0)
y=torch.tensor(0.0)

w=torch.tensor(1.0)

b=torch.tensor(0.0)

##BCE

def binary_cross(pred,target):
    
    # Small value to avoid numerical problems
    # Why? Because log(0) is undefined (-infinity)
    # So we never want prediction to become exactly 0 or exactly 1
    epsilon = 1e-8

    # Clamp means: force values to stay inside a safe range
    # If prediction < epsilon → set it to epsilon
    # If prediction > (1 - epsilon) → set it to (1 - epsilon)
    # This prevents log(0) when computing:
    # log(prediction) or log(1 - prediction)
    pred=torch.clamp(pred,epsilon,1.0-epsilon)

    loss = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))

    return loss


## Forward Pass

z=w*x+b

y_pred=torch.sigmoid(z)

loss=binary_cross(y_pred,y)
print(f"Loss: {loss}")
##Derivatives

# 1. dL/d(y_pred): Loss with respect to the prediction (y_pred)
dloss_dy_pred = (y_pred - y)/(y_pred*(1-y_pred))

# 2. dy_pred/dz: Prediction (y_pred) with respect to z (sigmoid derivative)
dy_pred_dz = y_pred * (1 - y_pred)

# 3. dz/dw and dz/db: z with respect to w and b
dz_dw = x  # dz/dw = x
dz_db = 1  # dz/db = 1 (bias contributes directly to z)

dL_dw = dloss_dy_pred * dy_pred_dz * dz_dw
dL_db = dloss_dy_pred * dy_pred_dz * dz_db

print(f"Manual Gradient of loss  weight (dw): {dL_dw}")
print(f"Manual Gradient of loss  bias (db): {dL_db}")

"""
So first we computed the loss using binary cross-entropy, 
then we calculated the gradients of the loss with respect to the weight and bias manually. 
The gradients are essential for updating the parameters during training in a neural network.
"""