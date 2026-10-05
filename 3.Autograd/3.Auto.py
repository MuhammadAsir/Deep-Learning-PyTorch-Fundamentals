import torch


x = torch.tensor(6.7)
y = torch.tensor(0.0)


w=torch.tensor(1.0, requires_grad=True)
b=torch.tensor(0.0, requires_grad=True)

def binary_cross_entropy_loss(prediction, target):

    epsilon = 1e-8

    prediction = torch.clamp(prediction, epsilon, 1 - epsilon) 
 
    loss = -(target * torch.log(prediction) +
             (1 - target) * torch.log(1 - prediction))

    return loss

z = w*x + b
print(f"z: {z}")    

y_pred = torch.sigmoid(z)
print("Predicted value (y_pred):", y_pred)

loss=binary_cross_entropy_loss(y_pred, y)
print(f"Loss: {loss}")  

loss.backward()#Starting from the loss, we compute the gradients of all tensors that have requires_grad=True
print("Weights Gradient (dw):", w.grad) #dl/dw
print("Bias Gradient (db):", b.grad) #dl/db


##For multiple input

a=torch.tensor([1.0,2.0,3.0],requires_grad=True)

b=(a**2).mean()

b.backward()
print("Gradient of b with respect to a :", a.grad)  # db/da