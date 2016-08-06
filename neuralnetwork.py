#!/usr/bin/env python
# ex: expandtab ts=4 sts=4 sw=4:

import random
import gzip
import cPickle
import os.path

import numpy as np

# If the below is set to false and a file with name as the value of `dumpfilename` is present, will load the theta from that file instead of learning
ENABLE_LEARNING = True  
dumpfilename = 'nn.dat'

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.num_layers = len(layers)
        self.theta = [np.random.randn(j,i+1) for i,j in zip(layers[:-1], layers[1:]) ]

    def learn(self, X, y, epochs, alpha):
        self.__stochastic_gradient_descent(X, y, epochs, alpha)

    def predict(self, X):
        return self.__feed_forward(X)

    def __feed_forward(self, X):
        m, n = X.shape
        a = np.ones( (m, n+1) )
        a[:, 1:] = X
        for theta in self.theta:
            z = a.dot(theta.T)
            a = np.ones( (m, theta.shape[0] + 1) )
            a[:, 1:] = sigmoid(z)
        return a[:, 1:]  # No bias unit needed for the output layer

    def __stochastic_gradient_descent(self, xval, yval, epochs, alpha):
        m, n = xval.shape
        k = self.layers[-1]
        Y = np.eye(k)[yval]
        X = np.ones((m, n+1))
        X[:, 1:] = xval
        for __ in xrange(epochs):
            indices = random.sample(range(m), m)
            for xx, yy in zip(X[indices], Y[indices]):
                self.__update_theta(xx, yy, alpha)

    def __update_theta(self, x, y, alpha):
        grad_theta = self.__back_propogation(x, y)
        self.theta = [ theta - alpha*grad for theta, grad in zip(self.theta, grad_theta) ]

    def __back_propogation(self, x, y):
        z = [None]
        a = [x] 
        for l in xrange(self.num_layers-1):
            cur_z = np.dot( self.theta[l], a[l] )
            cur_a = np.insert( sigmoid(cur_z), 0, 1 )  # Inserted the bias unit as the first element
            z.append(cur_z)
            a.append(cur_a)
        a[-1] = a[-1][1:]
        delta = [ np.zeros(wl) for wl in self.layers ]
        delta[-1] = a[-1] - y
        for l in reversed(range(1,self.num_layers-1)):
            delta[l] = np.dot( self.theta[l][:,1:].T, delta[l+1] ) * sigmoid_prime(z[l])
        grad = [ np.zeros(theta.shape) for theta in self.theta ]
        for l in xrange(self.num_layers-1):
            grad[l] = np.dot( delta[l+1].reshape(-1,1), a[l].reshape(-1,1).T )
        return grad


def sigmoid(z):
    return 1.0 / ( 1.0 + np.exp(-z) )

def sigmoid_prime(z):
    return sigmoid(z) * ( 1 - sigmoid(z) )

def load_data():
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        training_data, validation_data, test_data = cPickle.load(f)
    tr_x = training_data[0]
    va_x = validation_data[0]
    te_x = test_data[0]
    tr_y = training_data[1]
    va_y = validation_data[1]
    te_y = test_data[1]

    return (tr_x, tr_y, va_x, va_y, te_x, te_y)

def calculate_accuracy(nnoutput, expected):
    prediction = map(np.argmax, nnoutput)
    correct = 0
    for pred, actual in zip( prediction, expected ):
        correct += pred == actual
    return float(correct)/len(expected) * 100

def main():
    train_x, train_y, valid_x, valid_y, test_x, test_y = load_data()
    epochs = 20
    alpha = 0.1
    if ENABLE_LEARNING or not os.path.isfile(dumpfilename):
        nn = NeuralNetwork([784, 30, 10])
        nn.learn(train_x, train_y, epochs, alpha)
        print '--- Learning Complete ---'
        print '--- Dumping data to file nn.dat ---'
        with open(dumpfilename, 'wb') as fd:
            cPickle.dump(nn, fd)
    else:
        with open(dumpfilename) as fd:
            nn = cPickle.load(fd)
    # Calculating training set accuracy
    result = nn.predict(train_x)
    print 'Training Set Accuracy: ', calculate_accuracy(result, train_y), '%'
    # Calculating validation set accuracy
    result = nn.predict(valid_x)
    print 'Validation Set Accuracy:', calculate_accuracy(result, valid_y), '%'
    # Calculating test set accuracy
    result = nn.predict(test_x)
    print 'Test Set Accuracy:', calculate_accuracy(result, test_y), '%'


if __name__ == '__main__':
    main()

