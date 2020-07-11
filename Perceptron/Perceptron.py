import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
     # Constructor method is defined to assign number of epochs denoted by epoch and learning rate denoted by eta
  def __init__(self, epochs = 100, eta = 1):
    self.epochs = epochs
    self.eta = eta
    
    # Train perceptron. Parameters are the object, points and matching labels.
  def train(self, X, y):
    # get the number of features of the dataset. X.shape returns the size of the matrix as [mxn]. We are looking for the number n
    num_features = X.shape[1]
    # define a weight array depending on an object. This array length should be the number of features of the dataset+1. +1 comes from the bias weight.
    # fill the first w array with zeros.
    self.w = np.zeros(num_features + 1)
    # hold a counter to count the number of times that the algorithm takes to converge. Initialize with zero.
    self.counter=0
    # create a val variable. Assign self.w[0] value to val variable.
    # the point of doing this: In the for loop below, self.w[0] value is updated by weight update value. Weight update is only change if the current
    # separator line of the perceptron is not working correctly and needs to be updated. When weight update value changes so does self.w[0] change. 
    # So in the each epoch I compare the val with self.w[0] to see if there is a change. If there is a difference then counter is incremented by 1
    val=self.w[0]
    # Perform the epochs
    for i in range(self.epochs):
        # For every combination of X and y
        # check if self.w[0] is changed. If so increment the counter by 1. This keep tracks of the number of iterations it takes to converge.
        if val!=self.w[0]:
            self.counter=self.counter+1
        # Update val 
        val=self.w[0]
        # look into the samples and outcomes of combined X and y
        for sample, outcome in zip(X, y):
        # Turning -1 labels to 0 like a binary classification.
            if (outcome==-1):
                outcome=0
            # Use the predict method to generate a prediction and compare it with the outcome
            prediction    = self.predict(sample)
            # difference between prediction and actual outcome
            diff    = (outcome - prediction)
            # Weight update = w_update variable is created. The value of the variable calculated as in PLA rule. Learning rate*difference
            w_update = self.eta * diff
            # update the parameters of weight vector accordingly. If there is no difference then the weight vector should not change.
            self.w[1:]    += w_update * sample
            self.w[0]     += w_update
    
    # returns the object    
    return self

  # Generates a prediction based on the given sample
  def predict(self, sample):
    # np.dot calculates the dot product of given arrays
    # outcome is the added value of dot prodcut and weight update
    outcome = np.dot(sample, self.w[1:]) + self.w[0]
    # a prediction value is returned depending on the value of outcome. If it is positive prediction is 1, if it is negative prediction is zero
    return np.where(outcome > 0, 1, 0)