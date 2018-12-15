"""
==============

KATHERYN HRABIK
CSC578-710  (ONLINE)
HOMEWORK 1 - APPLICATION CODE

==============
"""


##Loading iris.dat to check MSE, cross-entropy, loglikelihood implementations

import NN578_network as network

import numpy as np

iris_test_net = network.load_network("iris-423.dat")


##Creating the same two instances as shown in the example

inst1 = (np.array([5.7, 3, 4.2, 1.2]), np.array([0., 1., 0.]))

x1 = np.reshape(inst1[0], (4, 1))
y1 = np.reshape(inst1[1], (3, 1))

sample1 = [(x1, y1)]

inst2 = (np.array([4.8, 3.4, 1.6, 0.2]), np.array([1.,  0., 0.]))

x2 = np.reshape(inst2[0], (4, 1))
y2 = np.reshape(inst2[1], (3, 1))

sample2 = [(x2, y2)]


##Calling SGD: one instance for training, one for testing.
##Running network for just one epoch, with eta = 1.0 and mini_batch_size = 1

print("IRIS.DAT - MSE, CROSSENTROPY, LOGLIKELIHOOD VERIFICATION")
print("\n")
iris_test_net.SGD(sample1, 1, 1, 1.0, sample2)


##Loading XOR.csv


import NN578_network as network

import numpy as np

ret = np.genfromtxt('../data/xor.csv', delimiter=',')

temp = np.array([(entry[:2],entry[2:]) for entry in ret])

temp_inputs = [np.reshape(x, (2, 1)) for x in temp[:,0]]
temp_results = [network.vectorize_target(2, y) for y in temp[:,1]]

xor_data = list(zip(temp_inputs, temp_results))

##Create a network of the size [2, 4, 2] - i.e. one hidden layer with 4 nodes


xor_test_net = network.Network([2, 4, 2])

##Train using max_epochs = 30, mini_batch_size = 1, and eta = 2.2
print("XOR CSV TRIAL 1")
print("\n")
xor_test_net.SGD(xor_data, 30, 1, 2.2)
print("XOR CSV TRIAL 2")
print("\n")
xor_test_net.SGD(xor_data, 30, 1, 2.2)
print("XOR CSV TRIAL 3")
print("\n")
xor_test_net.SGD(xor_data, 30, 1, 2.2)

##Use a slightly larger dataset, iris.csv, for an open experiment

import NN578_network as network
import numpy as np

ret = np.genfromtxt('../data/iris.csv', delimiter=',')
temp = np.array([(entry[:4],entry[4:]) for entry in ret])

temp_inputs = [np.reshape(x, (4, 1)) for x in temp[:,0]]
temp_results = [np.reshape(y, (3, 1)) for y in temp[:,1]] 
iris_data = list(zip(temp_inputs, temp_results))


##Iris CSV Network Examples

iris_csv_net = network.Network([4, 3, 3, 3])
print("IRIS CSV EXPERIMENT 1")
print("\n")
iris_csv_net.SGD(iris_data, 30, 10, .5)


iris_csv_net = network.Network([4, 200, 150, 150, 150, 50, 35, 50, 60, 75, 25, 10, 3])
print("IRIS CSV EXPERIMENT 2")
print("\n")
iris_csv_net.SGD(iris_data, 30, 10, .5)


iris_csv_net = network.Network([4, 3, 3, 3])
print("IRIS CSV EXPERIMENT 3")
print("\n")
iris_csv_net.SGD(iris_data, 1000, 10, .5)



