
1. Introduction<br /> 
Your friends now want you to try implementing a neural network to classify MNIST digits.<br /> 

Setup:<br /> 

As with the last project, please use Python's NumPy numerical library for handling arrays and array operations; use matplotlib for producing figures and plots.<br /> 

Note on software: For all the projects, we will use python 3.6 augmented with the NumPy numerical toolbox, the matplotlib plotting toolbox. For THIS project, you will also be using PyTorch for implementing the Neural Nets and scipy to handle sparse matrices.<br /> 

Download mnist.tar.gz and untar it in to a working directory. The archive contains the various data files in the Dataset directory, along with the following python files:<br /> 

part2-nn/neural_nets.py in which you'll implement your first neural net from scratch<br /> 
part2-mnist/nnet_fc.py where you'll start using PyTorch to classify MNIST digits<br /> 
part2-mnist/nnet_conv.py where you will use convolutional layers to boost performance<br /> 
part2-twodigit/mlp.py and part2-twodigit/conv.py which are for a new, more difficult version of the MNIST dataset<br /> 

Tip: Throughout the whole online grading system, you can assume the NumPy python library is already imported as np. In some problems you will also have access to python's random library, and other functions you've already implemented. Look out for the "Available Functions" Tip before the codebox, as you did in the last project.<br /> 

This project will unfold both on MITx and on your local machine. However, we encourage you to first implement the functions locally. For this project, there will not be a test.py script. You are encouraged to think of your own test cases to make sure your code works as you expected before submitting it to the online grader.<br /> 


2. Neural Network Basics<br /> 
Good programmers can use neural nets. Great programmers can make them. This section will guide you through the implementation of a simple neural net with an architecture as shown in the figure below. You will implement the net from scratch (you will probably never do this again, don't worry) so that you later feel confident about using libraries. We provide some skeleton code in neural_nets.py for you to fill in.<br /> 
