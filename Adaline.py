import numpy as np

"""
    Class that implements an adaptive linear neuron machine learning
    algorithm.

    Weights for each elements are changed after each iteration to minimized
    the cost function via gradient descent.

    """
class Adaline(object):

    def __init__(self, lrate = 0.01, numiter = 50):
        self.lrate = lrate
        self.numiter = numiter

    def fit(self, X, y):

        self.weights_ = np.zeros(1+X.shape[1]) #weights are all initialized to zero
        self.cost_ = []

        for i in range(self.numiter): #for each iteration 
            result = self.net_input(X) #get the list of predicted values for "object" in X
            errors = (y-result) #get the list of errors from the expected values
            self.weights_[1:] += self.lrate * X.T.dot(errors) #update weights 
            self.weights_[0] += self.lrate * errors.sum()

            cost = (errors**2).sum()/2.0 #calculate the cost and append to the cost_ list for tracking purposes
            self.cost_.append(cost)

            return self

    def net_input(self,X):
        return np.dot(X, self.weights_[1:]) + self.weights[0] #the net input is equal to the dot product of the X matrix and the weights list
    

    def activation(self,X): #activation function to assign class labels
        return self.net_input(X)

    def predict(self,X): #predict function uses activation function to assign class labels
        return np.where(self.activation(X) >= 0.0, 1, -1)
