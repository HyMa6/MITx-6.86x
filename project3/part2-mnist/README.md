7. Classification for MNIST using deep neural networks<br /> 
In this section, we are going to use deep neural networks to perform the same classification task as in previous sections. We will use PyTorch, a python deep learning framework. Using a framework like PyTorch means you don't have to implement all of the details (like in the earlier problem) and can spend more time thinking through your high level architecture.

Setup Overview To setup PyTorch, navigate to their website in your browser, select your preferences and begin downloading. Your selection for OS and Package Manager will depend on your local setup. For example, if you are on a Mac and use pip as your Python package manager, select "OSX" and "Pip". We recommend you select Python version 3 for use with PyTorch. Finally, you are not required to train large models for this course, so you can safely select "None" for CUDA. If you have access to a NVIDIA GPU enabled device with the CUDA library installed, and want to try training your neural models on GPUs, feel free to install PyTorch with CUDA selected but you will have to troubleshoot on your own.

Test your installation Once you have successfully installed PyTorch using the instructions on their website, you should test your installation to ensure it is running properly before trying to complete the project. For basic functionality, you can start a python REPL environment with the python command in your terminal. Then try importing PyTorch with import torch.
<br /> 


8. Fully-Connected Neural Networks<br /> 
First, we will employ the most basic form of a deep neural network, in which the neurons in adjacent layers are fully connected to one another.

You will be working in the filespart2-mnist/nnet_fc.pyin this problem

Training and Testing Accuracy Over Time
We have provided a toy example nnet_fc.py in which we have implemented for you a simple neural network. This network has one hidden layer of 10 neurons with a rectified linear unit (ReLU) nonlinearity, as well as an output layer of 10 neurons (one for each digit class). Finally, a softmax function normalizes the activations of the output neurons so that they specify a probability distribution. Reference the PyTorch Documentation and read through it in order to gain a better understanding of the code. Then, try running the code on your computer with the command python3 nnet_fc.py. This will train the network with 10 epochs, where an epoch is a complete pass through the training dataset. Total training time of your network should take no more than a couple of minutes. At the end of training, your model should have an accuracy of more than %85 on test data.

Note: We are not using a softmax layer because it is already present in the loss: PyTorch's nn.CrossEntropyLoss combines nn.LogSoftMax with nn.NLLLoss.
<br /> 
Improving Accuracy<br /> 

We would like to try to improve the performance of the model by performing a mini grid search over hyper parameters (note that a full grid search should include more values and combinations). To this end, we will use our baseline model (batch size 32, hidden size 10, learning rate 0.1, momentum 0 and the ReLU activation function) and modify one parameter each time while keeping all others to the baseline. We will use the validation accuracy of the model after training for 10 epochs. For the LeakyReLU activation function, use the default parameters from pyTorch (negative_slope=0.01).

Note: If you run the model multiple times from the same script, make sure to initialize the numpy and pytorch random seeds to 12321 before each run.

Which of the following modifications achieved the highest validation accuracy?

<br /> 
baseline (no modifications)
batch size 64
learning rate 0.01
momentum 0.9
LeakyReLU activation
<br /> 

Does the model variation that achieved the highest validation accuracy achieved also the highest test accuracy?
<br /> 

Yes<br /> 

Improving Accuracy <br /> 
Modifying the model's architecture is also worth considering. Increase the hidden representation size from 10 to 128 and repeat the grid search over the hyper parameters. This time, what modification achieved the highest validation accuracy?
-LeakyreLU activation


