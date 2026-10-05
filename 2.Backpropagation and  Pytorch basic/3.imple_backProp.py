import torch


data=torch.tensor([

    [2,3,3],
    [5,6,4],
    [7,8,5]
],dtype=torch.float32)

x=data[:,0:2]
y=data[:,2]

#We don't initialize all neural-network weights to 0 because then all neurons learn the same thing.

def initialize_parameters():
    parameters={}

    parameters['W1']=torch.ones((2,2))*0.1
    parameters['B1']=torch.zeros((2,1))

    parameters['W2']=torch.ones((2,1))*0.1
    parameters['B2']=torch.zeros((1,1))

    return parameters

print("Parameters initialized:",'\n', initialize_parameters())

##Better way

def initialize_parameters_better(layer_dim):
    
    torch.manual_seed(3) 
    parameters={}
    loop=len(layer_dim)

    for l in range(1,loop):
        parameters['W'+str(l)]=torch.ones((layer_dim[l-1],layer_dim[l]))*0.1
        parameters['B'+str(l)]=torch.zeros((layer_dim[l],1))
    #(layer_dim[l-1],layer_dim[l]), means that the weight matrix W for layer l has dimensions corresponding to the number of neurons in the previous layer (layer_dim[l-1]) and the number of neurons in the current layer (layer_dim[l]). This is important for ensuring that the matrix multiplication during forward propagation works correctly.
    
    return parameters
print('\n')
print("Better Way Parameters initialized:",'\n', initialize_parameters_better([2,2,1]))

##Forward Propagation
def linear_forward(A_prev,W,B):
    Z=torch.matmul(W.T,A_prev)+B

    return Z

#We use linear_forward function to compute the linear part of the forward propagation step in a neural network. It takes the activations from the previous layer (A_prev), the weights (W), and the biases (B) as inputs, and computes the linear combination Z = W^T * A_prev + B. This Z is then used as input to an activation function in the next step of forward propagation.

def L_layer_forward(x,parameters):
    A=x
    loop=len(parameters)//2 #We divide by 2 because we have both weights and biases for each layer, so the total number of parameters is twice the number of layers.

    for l in range(1,loop+1): #loop+1,means that the loop will iterate from 1 to loop (inclusive), allowing us to access the parameters for each layer correctly.
            
            A_prev=A
            W=parameters['W'+str(l)]
            B=parameters['B'+str(l)]
            A=linear_forward(A_prev,W,B)
            
#A is storing the activations from the current layer, while A_prev is storing the activations from the previous layer. This is important because during backpropagation, we will need to use A_prev to compute gradients and update the weights and biases accordingly. By keeping track of both A and A_prev, we can ensure that we have the necessary information for both forward and backward passes in the neural network.
            
    return A,A_prev 
#Why A_prev? Because we need to keep track of the activations from the previous layer for backpropagation. During backpropagation, we will use these previous activations to compute gradients and update the weights and biases accordingly.


x_sample=x[0].reshape(2,1)#reshape it because we want to treat it as a column vector for matrix multiplication in the forward propagation step.
params=initialize_parameters_better([2,2,1])
y_hat,A_prev=L_layer_forward(x_sample,params)

#y_hat will get current layer output and A_prev will get previous layer output. We need both for backpropagation.

print('\n') 
print("Forward Propagation Output:",'\n', y_hat)
print("Previous Layer Output:",'\n', A_prev)


##Update weights and biases using backpropagation. 

def update_parameters(parameters, y, y_hat, A1, X, lr=0.001):

    err_signal = 2 * (y - y_hat)

     #  Save OLD W2 values before updating
    W2_00_old = parameters['W2'][0, 0].item()
    W2_10_old = parameters['W2'][1, 0].item()

    # Update Layer 2




    parameters['W2'][0, 0] += lr * err_signal * A1[0, 0]
    parameters['W2'][1, 0] += lr * err_signal * A1[1, 0]
    parameters['B2'][0, 0] += lr * err_signal #


    parameters['W1'][0, 0] += lr * err_signal * parameters['W2'][0, 0] * X[0, 0]
    parameters['W1'][0, 1] += lr * err_signal * parameters['W2'][0, 0] * X[1, 0]
    parameters['B1'][0, 0] += lr * err_signal * parameters['W2'][0, 0]

    parameters['W1'][1, 0] += lr * err_signal * parameters['W2'][1, 0] * X[0, 0]
    parameters['W1'][1, 1] += lr * err_signal * parameters['W2'][1, 0] * X[1, 0]
    parameters['B1'][1, 0] += lr * err_signal * parameters['W2'][1, 0]

    return parameters


# using loop



epochs = 5

for i in range(epochs):
    epoch_loss = 0
    for j in range(x.shape[0]):
        # Prepare sample
        X_sample = x[j].reshape(2, 1)
        y_sample = y[j]

        # Forward
        y_hat, A1 = L_layer_forward(X_sample, params)
        y_hat_scalar = y_hat[0, 0]

        # Update
        params = update_parameters(params, y_sample, y_hat_scalar, A1, X_sample)

        # Loss calculation
        loss = (y_sample - y_hat_scalar) ** 2
        epoch_loss += loss.item()

    print(f"Epoch - {i+1} Loss - {epoch_loss / x.shape[0]}")