import torch

X=torch.tensor([[1,2,3],[4,5,6]])

X=X+2 #Doing addition
print(X)

X=X*3
print(X)

a=torch.rand(2,3)
b=torch.rand(2,3)
print(a+b,'\n',a-b,'\n',a*b,'\n',a/b)


c=torch.tensor([[-1,-2,-3],[-4,-5,-6]])
torch.abs(c)#output: tensor([[1, 2, 3],[4, 5, 6]])

d=torch.tensor([[1,2,3],[4,5,6]])
torch.neg(d)#output: tensor([[-1, -2, -3],[-4, -5, -6]])
torch.sqrt(d)#output: tensor([[1.0000, 1.4142, 1.7321],[2.0000, 2.2361, 2.4495]])

#clamp function is used to limit the values of a tensor within a specified range. It takes three arguments: the input tensor, the minimum value, and the maximum value. Any value in the input tensor that is less than the minimum value will be set to the minimum value, and any value that is greater than the maximum value will be set to the maximum value. The output tensor will have the same shape as the input tensor.
torch.clamp(d,min=2,max=5)#output: tensor([[2, 2, 3],[4, 5, 5]])

##Reduction Operation
e=torch.randint(size=(2,3),low=0,high=10,dtype=torch.float32)
torch.sum(e)#output: tensor(24, dtype=torch.int32)
torch.sum(e,dim=0)#output: tensor([ 9,  7,  8], dtype=torch.int32),column-wise sum
torch.sum(e,dim=1)#output: tensor([ 9, 15], dtype=torch.int32),row-wise sum
print("Mean of e: ",torch.mean(e))
torch.median(e)
torch.max(e)
print("Product of e: ",torch.prod(e))#product of all elements in the tensor

print("Argmax of e: ",torch.argmax(e))#returns the index of the maximum value in the tensor
print("Argmin of e: ",torch.argmin(e))#returns the index of the minimum value in the tensor 

##Matrix Operation

f=torch.randint(size=(2,3),low=0,high=10)
g=torch.randint(size=(3,2),low=0,high=10)

print(torch.matmul(f,g))#Matrix multiplication of f and g


vector=torch.tensor([1,2,3])
vector2=torch.tensor([4,5,6])
print(torch.dot(vector,vector2))#Dot product of two vectors

print("Transpose of f: ",torch.transpose(f,0,1))#transpose of f matrix, 0 and 1 are the dimensions to be swapped

h=torch.randint(size=(3,3),low=1,high=10,dtype=torch.float32) 
print("determinant of h: ",torch.det(h.float()))#determinant of h matrix, we need to convert it to float because determinant is not defined for integer matrices    

##Comparison Operators

i=torch.randint(size=(2,3),low=0,high=10)
j=torch.randint(size=(2,3),low=0,high=10)
print(i>j,'\n',i<j,'\n',i==j,'\n',i!=j)#comparison operators, returns a tensor of boolean values


##Special function

k=torch.tensor([[10.0,20.0,30.0]])

torch.log(k)
torch.exp(k)
torch.sigmoid(k)
torch.softmax(k,dim=1)#softmax function, returns a tensor of the same shape as k with values between 0 and 1 that sum to 1 along the specified dimension
torch.relu(k)#relu function, returns a tensor of the same shape as k with all negative values replaced by 0

##inplace operation.It helps to save memory by modifying the tensor in place instead of creating a new tensor. Inplace operations are denoted by an underscore (_) at the end of the function name. For example, x.add_(y) will add y to x in place, modifying the values of x directly. Inplace operations can be more efficient than creating new tensors, especially for large tensors, but they can also lead to unexpected behavior if not used carefully. It is important to be aware of when inplace operations are being used and to avoid modifying tensors that are needed for other computations.

m=torch.rand(2,3)
n=torch.rand(2,3)

print("Before inplace operation: ",m)
print("After inplace operation: ",m.add_(n))#inplace addition of n to m, modifies the values of m directly
#That's the difference between inplace operation and normal operation. In normal operation, we create a new tensor to store the result of the operation, whereas in inplace operation, we modify the values of the original tensor directly.

print(torch.relu_(m))#inplace relu operation, modifies the values of m directly

##copying tensor


a=torch.tensor([[1,2,3,4,5,6,7]])

b=a.clone()#creates a copy of a tensor, b is a new tensor with the same values as a
b[0][2]=100
print(a,'\n',b)#a is not modified, b is modified