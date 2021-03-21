The goal of this project is to design a classifier to use for sentiment analysis of product reviews. Our training set consists of reviews written by Amazon customers for various food products. The reviews, originally given on a 5 point scale, have been adjusted to a +1 or -1 scale, representing a positive or negative review, respectively. 

Below are two example entries from our dataset. Each entry consists of the review and its label. The two reviews were written by different customers describing their experience with a sugar-free candy.

Review	label
Nasty No flavor. The candy is just red, No flavor. Just plan and chewy. I would never buy them again	 −1 
YUMMY! You would never guess that they're sugar-free and it's so great that you can eat them pretty much guilt free! i was so impressed that i've ordered some for myself (w dark chocolate) to take to the office. These are just EXCELLENT!	 1 

In order to automatically analyze reviews, you will need to complete the following tasks:

Implement and compare three types of linear classifiers: the perceptron algorithm, the average perceptron algorithm, and the Pegasos algorithm.

Use your classifiers on the food review dataset, using some simple text features.

Experiment with additional features and explore their impact on classifier performance.

Setup Details:


project1.py contains various useful functions and function templates that you will use to implement your learning algorithms.

main.py is a script skeleton where these functions are called and you can run your experiments.

utils.py contains utility functions that the staff has implemented for you.

test.py is a script which runs tests on a few of the methods you will implement. 

Hinge Loss
In this project you will be implementing linear classifiers beginning with the Perceptron algorithm. You will begin by writing your loss function, a hinge-loss function. For this function you are given the parameters of your model  𝜃  and  𝜃0 . Additionally, you are given a feature matrix in which the rows are feature vectors and the columns are individual features, and a vector of labels representing the actual sentiment of the corresponding feature vector.

Perceptron Algorithm

Pegasos update rule
The following pseudo-code describes the Pegasos update rule.

Pegasos update rule (𝑥(𝑖),𝑦(𝑖),𝜆,𝜂,𝜃): 
if  𝑦(𝑖)(𝜃⋅𝑥(𝑖))≤1  then
update  𝜃=(1−𝜂𝜆)𝜃+𝜂𝑦(𝑖)𝑥(𝑖)  
else: 
update  𝜃=(1−𝜂𝜆)𝜃  

The  𝜂  parameter is a decaying factor that will decrease over time. The  𝜆  parameter is a regularizing parameter.

In this problem, you will need to adapt this update rule to add a bias term ( 𝜃0 ) to the hypothesis, but take care not to penalize the magnitude of  𝜃0 .

Algorithm Discussion
Once you have completed the implementation of the 3 learning algorithms, you should qualitatively verify your implementations. In main.py we have included a block of code that you should uncomment. This code loads a 2D dataset from toy_data.txt, and trains your models using  𝑇=10,𝜆=0.2 . main.py will compute  𝜃  and  𝜃0  for each of the learning algorithms that you have written. Then, it will call plot_toy_data to plot the resulting model and boundary.

Automative review analyzer
Now that you have verified the correctness of your implementations, you are ready to tackle the main task of this project: building a classifier that labels reviews as positive or negative using text-based features and the linear classifiers that you implemented in the previous section!

The Data

The data consists of several reviews, each of which has been labeled with  −1  or  +1 , corresponding to a negative or positive review, respectively. The original data has been split into four files:

reviews_train.tsv (4000 examples)
reviews_validation.tsv (500 examples)
reviews_test.tsv (500 examples)

To get a feel for how the data looks, we suggest first opening the files with a text editor, spreadsheet program, or other scientific software package (like pandas).
Translating reviews to feature vectors

We will convert review texts into feature vectors using a bag of words approach. We start by compiling all the words that appear in a training set of reviews into a dictionary , thereby producing a list of  𝑑  unique words.


We can then transform each of the reviews into a feature vector of length  𝑑  by setting the  𝑖th  coordinate of the feature vector to  1  if the  𝑖th  word in the dictionary appears in the review, or  0  otherwise. For instance, consider two simple documents “Mary loves apples" and “Red apples". In this case, the dictionary is the set  {Mary;loves;apples;red} , and the documents are represented as  (1;1;1;0)  and  (0;0;1;1) .

A bag of words model can be easily expanded to include phrases of length  𝑚 . A unigram model is the case for which  𝑚=1 . In the example, the unigram dictionary would be  (Mary;loves;apples;red) . In the bigram case,  𝑚=2 , the dictionary is  (Mary loves;loves apples;Red apples) , and representations for each sample are  (1;1;0),(0;0;1) . In this section, you will only use the unigram word features. These functions are already implemented for you in the bag of words function.
In utils.py, we have supplied you with the load data function, which can be used to read the .tsv files and returns the labels and texts. We have also supplied you with the bag_of_words function in project1.py, which takes the raw data and returns dictionary of unigram words. The resulting dictionary is an input to extract_bow_feature_vectors which computes a feature matrix of ones and zeros that can be used as the input for the classification algorithms. Using the feature matrix and your implementation of learning algorithms from before, you will be able to compute  𝜃  and  𝜃0 .


Parameter Tuning
You finally have your algorithms up and running, and a way to measure performance! But, it's still unclear what values the hyperparameters like  𝑇  and  𝜆  should have. In this section, you'll tune these hyperparameters to maximize the performance of each model.

One way to tune your hyperparameters for any given Machine Learning algorithm is to perform a grid search over all the possible combinations of values. If your hyperparameters can be any real number, you will need to limit the search to some finite set of possible values for each hyperparameter. For efficiency reasons, often you might want to tune one individual parameter, keeping all others constant, and then move onto the next one; Compared to a full grid search there are many fewer possible combinations to check, and this is what you'll be doing for the questions below.

In main.py uncomment Problem 8 to run the staff-provided tuning algorithm from utils.py. For the purposes of this assignment, please try the following values for  𝑇 : [1, 5, 10, 15, 25, 50] and the following values for  𝜆  [0.001, 0.01, 0.1, 1, 10]. For pegasos algorithm, first fix  𝜆=0.01  to tune  𝑇 , and then use the best  𝑇  to tune  𝜆 

Feature Engineering
Frequently, the way the data is represented can have a significant impact on the performance of a machine learning method. Try to improve the performance of your best classifier by using different features. In this problem, we will practice two simple variants of the bag of words (BoW) representation.

