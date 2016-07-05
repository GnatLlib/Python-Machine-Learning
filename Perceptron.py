import numpy as np

# basic implementation of a binary perceptron learning model that takes in a matrix X as data and a list y of the corresponding expected
# values, and determines a binary classification. 
class Perceptron(object):

    def __init__(self, lrate=0.01, numiter=10):
        self.lrate = lrate   #learning rate 
        self.numiter = numiter #number of iterations through training data

    def fit(self, X, y):
        self.weights_ = np.zeros(1+X.shape[1]) #weights array is initialized as all zeros
        self.errors_ = [] #list to keep track of errors

        for _ in range(self.numiter):
            errors = 0
            for xi, target in zip(X,y):
                delta = self.lrate * (target - self.predict(xi))
                self.weights_[1:] += delta * xi
                self.weights_[0] += delta
                errors += int(update!= 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self,X):
        return np.dot(X, self.weights[1:]) + self.weights[0]



    def predict(self,X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


