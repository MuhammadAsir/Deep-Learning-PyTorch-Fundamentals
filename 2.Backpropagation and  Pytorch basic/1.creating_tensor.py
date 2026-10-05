"""
Pytorch is an open-source machine learning library based on the Torch library, 
used for applications such as computer vision and natural language processing, 
primarily developed by Facebook's AI Research lab (FAIR).

Pytorch can do computations on both CPU and GPU. It can also do automatic differentiation, 
which is useful for training neural networks.

It does pythonically and has a high level of flexibility and speed. It is used for 
CV,deep learning applications and natural language processing.
Pythonically means that it is designed to be intuitive and easy to use for Python developers,
and it follows the conventions and idioms of the Python programming language.

Pytorch uses CUDA (Compute Unified Device Architecture) for GPU acceleration.Because of this, 
it is faster than other libraries like TensorFlow and Keras.

CUDA: It is a parallel computing platform and application programming interface (API) model 
    created by Nvidia.It allows software developers and software engineers to use a 
    CUDA-enabled graphics processing unit (GPU) for 
    general purpose processing (an approach known as GPGPU, General-Purpose computing on Graphics Processing Units).

Tensorflow:It is an end-to-end open-source platform for machine learning. 
            It has a comprehensive, flexible ecosystem of tools, libraries, 
            and community resources that lets researchers push the state-of-the-art in ML, 
            and developers easily build and deploy ML-powered applications.

Keras:It is an open-source software library that provides a Python interface for artificial neural networks.
      It is a part of the TensorFlow project, and it acts as an interface for the TensorFlow library.

Pytorch VS Tensorflow: 1.Pytorch is more flexible and easier to use than Tensorflow.
                       2.Pytorch is more suitable for research and experimentation, while Tensorflow is more suitable for production and deployment.
                       3.Pytorch has a more intuitive and Pythonic interface, while Tensorflow has a more complex and verbose interface.
                       4.Tensorflow has better support for distributed training and deployment, while Pytorch has better support for dynamic computation graphs and debugging.
                       5.Pytorch has a more active and vibrant community, while Tensorflow has a more established and mature ecosystem.

So,now which one we should do?

Answer: It depends on your use case and requirements. If you are a researcher or a student who wants to experiment with new ideas and models, 
        Pytorch is a better choice. If you are a developer or an engineer who wants to build and deploy 
        ML applications, Tensorflow is a better choice.


"""


import torch


print(torch.__version__,'\n')#Checking what version of torch is installed


#Checking if GPU is available or not,because we need to mention it in research papers if we are using GPU for training the model
if torch.cuda.is_available():
        print('GPU is available')
        print(f"device name :{torch.cude.get_device_name(torch.cuda.current_device())}")

else:
        print('GPU is not available')


one=torch.ones(2,3)#creating a tensor of shape (2,3) with all elements as 1
print(one)

random=torch.rand(2,3)
print(random)

torch.manual_seed(42)#setting the seed for random number generation for reproducibility
random2=torch.rand(2,3)
print(random2)

#tensor is a multi-dimensional array that can be used to represent data in Pytorch. It is similar to a numpy array, but it can also be used on a GPU for faster computations. Tensors can have different shapes and data types, and they can be created from python lists, numpy arrays, or other tensors.
data=torch.tensor([
        [1,2,3],
        [4,5,6]
])#Creating a tensor from a python list


print(torch.eye(3))#Creating a 3x3 identity matrix      


torch.arange(0,20,2)#Creating a tensor with values from 0 to 20 with a step of 2

torch.linspace(0, 10, 5)#Creating a tensor with 5 values linearly spaced between 0 and 10
#output: tensor([ 0.0000,  2.5000,  5.0000,  7.5000, 10.0000])


##Tensor's Shape

x=torch.tensor([[1,2,3],[4,5,6]])
print(x.shape)#output: torch.Size([2, 3])

y=torch.empty_like(x)#Creating a tensor with the same shape as x but with uninitialized values

z=torch.ones_like(x)#Creating a tensor with the same shape as x but with all elements as 1

s=torch.rand_like(x,dtype=torch.float32)#Creating a tensor with the same shape as x but with random values between 0 and 1 and data type as float32
print(x,'\n',y,'\n',z,'\n',s)

##Data type
print(x.dtype)#output: torch.int64


m=torch.tensor([1,2,3],dtype=torch.float32)#Creating a tensor with data type as float32

m = m.to(torch.int32)#Changing the data type of the tensor to int32
