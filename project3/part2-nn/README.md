3. Activation Functions<br /> 

The first step is to design the activation function for each neuron. In this problem, we will initialize the network weights to 1, use ReLU for the activation function of the hidden layers, and use an identity function for the output neuron. The hidden layer has a bias but the output layer does not. Complete the helper functions in neural_networks.py, including rectified_linear_unit and rectified_linear_unit_derivative, for you to use in the NeuralNetwork class, and implement them below.

You will be working in the file part2-nn/neural_nets.py in this problem

Correction note (Nov 1): In the part2-nn/neural_nets.py, in the definition of Class NeuralNetwork(), the initialization of weights has now been changed to an initialization as float rather than int. You could either re-download the updated project release mnist.tar.gz, or change the corresponding lines in part2-nn/neural_nets.py to the following, where we have added decimal points to all numbers in the initialization:

class NeuralNetwork():<br /> 

    def __init__(self):

        # DO NOT CHANGE PARAMETERS (Initialized to floats instead of ints)
        self.input_to_hidden_weights = np.matrix('1. 1.; 1. 1.; 1. 1.')
        self.hidden_to_output_weights = np.matrix('1. 1. 1.')
        self.biases = np.matrix('0.; 0.; 0.')
Rectified Linear Unit

First implement the ReLu activation function, which computes the ReLu of a scalar.

Note: Your function does not need to handle a vectorized input

Available Functions: You have access to the NumPy python library as np

Taking the Derivative
Now implement its derivative so that we can properly run backpropagation when training the net. Note: we will consider the derivative at zero to have the same value as the derivative at all negative points.

Note: Your function does not need to handle a vectorized input

Available Functions: You have access to the NumPy python library as np<br /> 


4. Training the Network<br /> 
Forward propagation is simply the summation of the previous layer's output multiplied by the weight of each wire, while back-propagation works by computing the partial derivatives of the cost function with respect to every weight or bias in the network. In back propagation, the network gets better at minimizing the error and predicting the output of the data being used for training by incrementally updating their weights and biases using stochastic gradient descent.

We are trying to estimate a continuous-valued function, thus we will use squared loss as our cost function and an identity function as the output activation function. 𝑓(𝑥) is the activation function that is called on the input to our final layer output node, and 𝑎̂ is the predicted value, while 𝑦 is the actual value of the input.
<br /> 
𝐶=12(𝑦−𝑎̂)2
(6.1)
𝑓(𝑥)=𝑥
(6.2)<br /> 
When you're done implementing the function train (below and in your local repository), run the script and see if the errors are decreasing. If your errors are all under 0.15 after the last training iteration then you have implemented the neural network training correctly.

You'll notice that the train functin inherits from NeuralNetworkBase in the codebox below; this is done for grading purposes. In your local code, you implement the function directly in your Neural Network class all in one file. The rest of the code in NeuralNetworkBase is the same as in the original NeuralNetwork class you have locally.

In this problem, you will see the network weights are initialized to 1. This is a bad setting in practice, but we do so for simplicity and grading here.

You will be working in the file part2-nn/neural_nets.py in this problem

Implementing Train
Available Functions: You have access to the NumPy python library as np, rectified_linear_unit, output_layer_activation, rectified_linear_unit_derivative, and output_layer_activation_derivative

Note: Functions rectified_linear_unit, rectified_linear_unit_derivative, and output_layer_activation_derivative can only handle scalar input. You will need to use np.vectorize to use them.
<br /> 

5. Predicting the Test Data<br /> 
Now fill in the code for the function predict, which will use your trained neural network in order to label new data.

You will be working in the file part2-nn/neural_nets.py in this problem
<br /> 
Implementing Predict<br /> 
Available Functions: You have access to the NumPy python library as np, rectified_linear_unit and output_layer_activation

Note: Functions rectified_linear_unit_derivative, and output_layer_activation_derivative can only handle scalar input. You will need to use np.vectorize to use them

