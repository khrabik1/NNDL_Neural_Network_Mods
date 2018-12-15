# NNDL_Neural_Network_Mods
Modifications to Nielsen's Network.Py 

Homework Assignment 1
Depaul University 
CSC 578 Neural Networks and Deep Learning
Professor Noriko Tomuro
Fall 2018

Code Contents:
-	A module to implement the stochastic gradient descent 
	learning algorithm for a feedforward neural network.
	Gradients are calculated using backpropagation. 

Modifications To Network Code (NN578_network)

-	Edit the function evaluate() so that, in addition to 
	accuracy, it computes Mean Squared Error(MSE),
	cross-entropy and log-likelihood
	-	The function should return correct count, accuracy,
		MSE, cross-entropy and log-likelihood in a list
	-	Node that the target y for the iris and xor datasets
		provided for this homework is changed to an array (ndarray
		of shape (n,1)) from a single scaler in Nielsen's original
		code. 

-	Edit the SGD() function so that it does the following:
	-	Call Evaluate() for training_data, at the end of every
		epoch, and print the returned results. It should also
		call evaluate() for test_data is well if it is passed in
	-	Note that evaluate() is called exactly once for each data set
	-	Collect the performance results returned from evaluate()
		for all epochs for training_data and test_data into individual
		lists, like a history, and return the two lists in a list
		(to the caller of the function)
	-	Add *early stopping*, which terminates the epoch loop 
		prematurely if the classification accuracy for the training data
		became 100%.

-	Edit the function backprop() so that the local variable 'activations' is
	initially allocated with a structure which holds the activation value of ALL
	layers from the start, rather than the current code which starts
	with just the input layer 

Application code

-	Check your implementation for Modification 1 (evaluate()), 2a (SGD), and 3
	(activations in backprop())
	-	Add a line in the function backprop() that prints the values in the 
		'activations' array
	-	In application code:
		-	Load saved network 'iris-423.dat'
		- 	define sample data instances
		- 	run the network for one epoch with eta=1.0 and mini_batch=1

-	Check your implementation for Modification 2b(SGD return value) and 2c (early stopping)
	-	Using 'xor.csv', check if your early stopping is working correctly
	-	Then create a network of the size [2,4,2] and train it using 
		max_epochs=30, mini_batch=1, eta=2.2 to verify early stopping
	-	Next you verify the value returned from SGD(). 	
	-	Finally, use a slightly larger dataset, "iris.csv" for an open experiment
		to manipulate network architecture configurations
