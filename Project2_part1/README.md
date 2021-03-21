Alice, Bob, and Daniel are friends learning machine learning together. After watching a few lectures, they are very proud of having learned many useful tools, including linear and logistic regression, non-linear features, regularization, and kernel tricks. To see how these methods can be used to solve a real life problem, they decide to get their hands dirty with the famous digit recognition problem using the MNIST (Mixed National Institute of Standards and Technology) database.

Hearing that you are an excellent student in the MITx machine learning class with solid understanding of the material and great coding ability in Python, they decide to invite you to their team and help them with implementing these different algorithms.

The MNIST database contains binary images of handwritten digits commonly used to train image processing systems. The digits were collected from among Census Bureau employees and high school students. The database contains 60,000 training digits and 10,000 testing digits, all of which have been size-normalized and centered in a fixed-size image of 28 × 28 pixels. Many methods have been tested with this dataset and in this project, you will get a chance to experiment with the task of classifying these images into the correct digit using some of the methods you have learned so far.

2. Linear Regression with Closed Form Solution
After seeing the problem, your classmate Alice immediately argues that we can apply a linear regression model, as the labels are numbers from 0-9, very similar to the example we learned from Unit 1. Though being a little doubtful, you decide to have a try and start simple by using the raw pixel values of each image as features.

Alice wrote a skeleton code run_linear_regression_on_MNIST in main.py, but she needs your help to complete the code and make the model work. 
To solve the linear regression problem, you recall the linear regression has a closed form solution:

 	 𝜃=(𝑋𝑇𝑋+𝜆𝐼)−1𝑋𝑇𝑌 	 	 
where  𝐼  is the identity matrix.

3. Support Vector Machine
Bob thinks it is clearly not a regression problem, but a classification problem. He thinks that we can change it into a binary classification and use the support vector machine we learned in Lecture 4 to solve the problem. In order to do so, he suggests that we can build an one vs. rest model for every digit. For example, classifying the digits into two classes: 0 and not 0.

Bob wrote a function run_svm_one_vs_rest_on_MNIST where he changed the labels of digits 1-9 to 1 and keeps the label 0 for digit 0. He also found that sklearn package contains an SVM model that you can use directly. He gave you the link to this model and hopes you can tell him how to use that.


4. Multinomial (Softmax) Regression and Gradient Descent
Daniel suggests that instead of building ten models, we can expand a single logistic regression model into a multinomial regression and solve it with similar gradient descent algorithm.

The main function which you will call to run the code you will implement in this section is run_softmax_on_MNIST in main.py (already implemented). In the appendix at the bottom of this page, we describe a number of the methods that are already implemented for you in softmax.py that will be useful.

In order for the regression to work, you will need to implement three methods. Below we describe what the functions should do. We have included some test cases in test.py to help you verify that the methods you have implemented are behaving sensibly.

Write a function compute_probabilities that computes, for each data point  𝑥(𝑖) , the probability that  𝑥(𝑖)  is labeled as  𝑗  for  𝑗=0,1,…,𝑘−1 .

The softmax function  ℎ  for a particular vector  𝑥  requires computing

ℎ(𝑥)=1∑𝑘−1𝑗=0𝑒𝜃𝑗⋅𝑥/𝜏⎡⎣⎢⎢⎢⎢𝑒𝜃0⋅𝑥/𝜏𝑒𝜃1⋅𝑥/𝜏⋮𝑒𝜃𝑘−1⋅𝑥/𝜏⎤⎦⎥⎥⎥⎥, 
 
where  𝜏>0  is the temperature parameter . When computing the output probabilities (they should always be in the range  [0,1] ), the terms  𝑒𝜃𝑗⋅𝑥/𝜏  may be very large or very small, due to the use of the exponential function. This can cause numerical or overflow errors. To deal with this, we can simply subtract some fixed amount  𝑐  from each exponent to keep the resulting number from getting too large. Since

ℎ(𝑥)=𝑒−𝑐𝑒−𝑐∑𝑘−1𝑗=0𝑒𝜃𝑗⋅𝑥/𝜏⎡⎣⎢⎢⎢⎢𝑒𝜃0⋅𝑥/𝜏𝑒𝜃1⋅𝑥/𝜏⋮𝑒𝜃𝑘−1⋅𝑥/𝜏⎤⎦⎥⎥⎥⎥=1∑𝑘−1𝑗=0𝑒[𝜃𝑗⋅𝑥/𝜏]−𝑐⎡⎣⎢⎢⎢⎢𝑒[𝜃0⋅𝑥/𝜏]−𝑐𝑒[𝜃1⋅𝑥/𝜏]−𝑐⋮𝑒[𝜃𝑘−1⋅𝑥/𝜏]−𝑐⎤⎦⎥⎥⎥⎥,
 
subtracting some fixed amount  𝑐  from each exponent will not change the final probabilities. A suitable choice for this fixed amount is  𝑐=max𝑗𝜃𝑗⋅𝑥/𝜏 .


5. Temperature
Smaller temperature leads to less variance
   
6. Changing Labels
We now wish to classify the digits by their (mod 3) value, such that the new label  𝑦(𝑖)  of sample  𝑖  is the old  𝑦(𝑖)(mod3) . An example is provided in the next section. (Reminder: Return the temp_parameter to be 1 if you changed it for the last section)

7. Classification Using Manually Crafted Features
The performance of most learning algorithms depends heavily on the representation of the training data. In this section, we will try representing each image using different features in place of the raw pixel values. Subsequently, we will investigate how well our regression model from the previous section performs when fed different representations of the data.

Dimensionality Reduction via PCA

Principal Components Analysis (PCA) is the most popular method for linear dimension reduction of data and is widely used in data analysis. For an in-depth exposition see: https://online.stat.psu.edu/stat505/lesson/11.

Briefly, this method finds (orthogonal) directions of maximal variation in the data. By projecting an  𝑛×𝑑 dataset  𝑋  onto  𝑘≤𝑑  of these directions, we get a new dataset of lower dimension that reflects more variation in the original data than any other  𝑘 -dimensional linear projection of  𝑋 . By going through some linear algebra, it can be proven that these directions are equal to the  𝑘  eigenvectors corresponding to the  𝑘  largest eigenvalues of the covariance matrix  𝑋˜𝑇𝑋˜ , where  𝑋˜  is a centered version of our original data.

Remark: The best implementations of PCA actually use the Singular Value Decomposition of  𝑋˜  rather than the more straightforward approach outlined here, but these concepts are beyond the scope of this course.

Cubic Features

In this section, we will also work with a cubic feature mapping which maps an input vector  𝑥=[𝑥1,…,𝑥𝑑] into a new feature vector  𝜙(𝑥) , defined so that for any  𝑥,𝑥′∈ℝ𝑑 :

𝜙(𝑥)𝑇𝜙(𝑥′)=(𝑥𝑇𝑥′+1)3 

8. Dimensionality Reduction Using PCA
PCA finds (orthogonal) directions of maximal variation in the data. In this problem we're going to project our data onto the principal components and explore the effects on performance.
Note that to project a given  𝑛×𝑑  dataset  𝑋  into its  𝑘 -dimensional PCA representation, one can use matrix multiplication, after first centering  𝑋 :

𝑋˜𝑉 

where  𝑋˜  is the centered version of the original data  𝑋  using the mean learned from training data and  𝑉  is the  𝑑×𝑘  matrix whose columns are the top  𝑘  eigenvectors of  𝑋˜𝑇𝑋˜ . This is because the eigenvectors are of unit-norm, so there is no need to divide by their length.

9. Cubic Features
In this section, we will work with a cubic feature mapping which maps an input vector  𝑥=[𝑥1,…,𝑥𝑑]  into a new feature vector  𝜙(𝑥) , defined so that for any  𝑥,𝑥′∈ℝ𝑑 :

𝜙(𝑥)𝑇𝜙(𝑥′)=(𝑥𝑇𝑥′+1)3 
The cubic_features function in features.py is already implemented for you. That function can handle input with an arbitrary dimension and compute the corresponding features for the cubic Kernel. Note that here we don't leverage the kernel properties that allow us to do a more efficient computation with the kernel function (without computing the features themselves). Instead, here we do compute the cubic features explicitly and apply the PCA on the output features.

If we explicitly apply the cubic feature mapping to the original 784-dimensional raw pixel features, the resulting representation would be of massive dimensionality. Instead, we will apply the cubic feature mapping to the 10-dimensional PCA representation of our training data which we will have to calculate just as we calculated the 18-dimensional representation in the previous problem. After applying the cubic feature mapping to the PCA representations for both the train and test datasets, retrain the softmax regression model using these new features and report the resulting test set error below.

10. Kernel Methods
As you can see, implementing a direct mapping to the high-dimensional features is a lot of work (imagine using an even higher dimensional feature mapping.) This is where the kernel trick becomes useful.

Recall the kernel perceptron algorithm we learned in the lecture. The weights  𝜃  can be represented by a linear combination of features:

𝜃=∑𝑖=1𝑛𝛼(𝑖)𝑦(𝑖)𝜙(𝑥(𝑖)) 
 
In the softmax regression fomulation, we can also apply this representation of the weights:

𝜃𝑗=∑𝑖=1𝑛𝛼(𝑖)𝑗𝑦(𝑖)𝜙(𝑥(𝑖)). 
 
ℎ(𝑥)=1∑𝑘𝑗=1𝑒[𝜃𝑗⋅𝜙(𝑥)/𝜏]−𝑐⎡⎣⎢⎢⎢⎢𝑒[𝜃1⋅𝜙(𝑥)/𝜏]−𝑐𝑒[𝜃2⋅𝜙(𝑥)/𝜏]−𝑐⋮𝑒[𝜃𝑘⋅𝜙(𝑥)/𝜏]−𝑐⎤⎦⎥⎥⎥⎥ 
 
ℎ(𝑥)=1∑𝑘𝑗=1𝑒[∑𝑛𝑖=1𝛼(𝑖)𝑗𝑦(𝑖)𝜙(𝑥(𝑖))⋅𝜙(𝑥)/𝜏]−𝑐⎡⎣⎢⎢⎢⎢⎢𝑒[∑𝑛𝑖=1𝛼(𝑖)1𝑦(𝑖)𝜙(𝑥(𝑖))⋅𝜙(𝑥)/𝜏]−𝑐𝑒[∑𝑛𝑖=1𝛼(𝑖)2𝑦(𝑖)𝜙(𝑥(𝑖))⋅𝜙(𝑥)/𝜏]−𝑐⋮𝑒[∑𝑛𝑖=1𝛼(𝑖)𝑘𝑦(𝑖)𝜙(𝑥(𝑖))⋅𝜙(𝑥)/𝜏]−𝑐⎤⎦⎥⎥⎥⎥⎥ 
 
We actually do not need the real mapping  𝜙(𝑥) , but the inner product between two features after mapping:  𝜙(𝑥𝑖)⋅𝜙(𝑥) , where  𝑥𝑖  is a point in the training set and  𝑥  is the new data point for which we want to compute the probability. If we can create a kernel function  𝐾(𝑥,𝑦)=𝜙(𝑥)⋅𝜙(𝑦) , for any two points  𝑥  and  𝑦 , we can then kernelize our softmax regression algorithm.

In the last section, we explicitly created a cubic feature mapping. Now, suppose we want to map the features into d dimensional polynomial space,

𝜙(𝑥)=⟨𝑥2𝑑,…,𝑥21,2‾√𝑥𝑑𝑥𝑑−1,…,2‾√𝑥𝑑𝑥1,2‾√𝑥𝑑−1𝑥𝑑−2,…,2‾√𝑥𝑑−1𝑥1,…,2‾√𝑥2𝑥1,2𝑐‾‾√𝑥𝑑,…,2𝑐‾‾√𝑥1,𝑐⟩ 
 
Gaussian RBF Kernel
Another commonly used kernel is the Gaussian RBF kenel. Similarly, write a function rbf_kernel that takes in two matrices  𝑋  and  𝑌  and computes the RBF kernel  𝐾(𝑥,𝑦)  for every pair of rows  𝑥  in  𝑋  and  𝑦  in  𝑌 .




