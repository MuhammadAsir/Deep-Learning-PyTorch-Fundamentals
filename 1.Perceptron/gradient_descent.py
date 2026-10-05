import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression





class GDRegressor:
  def __init__(self,learning_rate,epochs):

    self.m = 1
    self.b = 0
    self.lr = learning_rate
    self.epochs = epochs

  def fit(self,X,y):
    #calculate b

    for i in range(self.epochs ):
      loss_slope_b = -2*np.sum( y - self.m*X.ravel() - self.b )

      loss_slope_m = -2 * np.sum((y - self.m*X.ravel() - self.b)*X.ravel())

      self.m = self.m - (self.lr * loss_slope_m)
      self.b = self.b - (self.lr*loss_slope_b)



      print(f" m = {self.m}, b = {self.b}")

  def predict(self,X):
    return self.m * X +self.b
  


X,y = make_regression(n_samples=100, n_features=1, n_informative=1, n_targets=1,noise=20,random_state=13)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
gd = GDRegressor(0.001,50)
gd.fit(X_train,y_train)