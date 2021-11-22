# LINEAR REGRESSION
## PROBLEM STATEMENT

## PART 1(A):
The objective here is to implement the concepts of regression learnt in class via polynomial curve fitting. To recap, polynomial curve fitting is an example of regression. In regression, the objective is to learn a function that maps an input variable x to a continuous target variable t.

For the first part of this assignment, we provide a personalised input file that contains data of the form:

````
x 1 ,t 1
x 2 ,t 2
.
.
.
x 100 ,t 100

````

The relation between x and t is of the form

````
t = w0 + w1x + ... + wMxM + ? 
````

where the noise (?) is drawn from a normal distribution with mean 0 and unknown (but fixed, for a given file) variance. M is also unknown.The end goal is to identify the underlying polynomial (both the degree and the coefficients), as well as to obtain an estimate of the noise variance.

The tasks to be accomplished are:

* By default, the implementation should carry out least-squares regression. The minimisation of the error function should be attempted in two ways: by directly computing the Moore-Penrose pseduoinverse (pinv), and via gradient descent (gd). For the latter, you should have a parameter controlling the batch size: ranging from 1 (stochastic gradient descent) to N (full batch gradient descent).

* To begin with, use only the first 20 data points in your file. (Note that you will need to generate the design matrix by creating feature vectors containing the powers of x from 1 to M.)

* Solve the curve fitting regression problem using error function minimisation – try the different minimisation approaches implemented above, and comment on any variation in the results, especially with change in batch size for gradient descent. Also explore how the convergence of gradient descent varies with the stopping criterion (you could plot the loss as a function of the number of iterations).

* You can define your own error function other than sum-of-squares error (note that the error function need not be convex). Try different error formulations and report the results. Also try and use a validation approach to characterise the goodness of fit for polynomials of different order. Can you distinguish overfitting, underfitting, and the best fit? In addition to this, obtain an estimate for the noise variance.

* Introduce regularisation and observe the changes. For quadratic regularisation, can you obtain an estimate of the optimal value for the regularisation parameter ? What is your corresponding best guess for the underlying polynomial? And the noise variance?

* Now repeat all of the above using the full data set of 100 points. How are your results affected by adding more data? Comment on the differences.

* At the end: what is your final estimate of the true underlying polynomial? Why?

## PART 1(B):
You are provided with a second personalised data set, available at https://web.iitd.ac.in/~sumeet/A1/2019MT60763/non_gaussian.csv. It is generated from a different polynomial, and the noise is now non-Gaussian. Can you repeat the analysis for this data set, focusing in particular on characterising the noise – can you figure out what kind of noise it is? Please justify your answer using appropriate analysis.

## PART 3:
For this part, you will be trying to model a real-world time-series data set. This contains measurements of a certain quantity at different points in time. (Details of what that quantity is will be revealed later.) The provided data sets should be downloaded from https://web.iitd.ac.in/~sumeet/A1/train.csv and https://web.iitd.ac.in/~sumeet/A1/test.csv. The train set contains 110 time points; and the test set contains another 10 points for which the measured value has been removed. In each row, the first value is the
date of the measurement (in US format, so month comes before day), and the second value is the actual measurement.Your task is to train a linear regression model (using your above implementation along with appropriate basis functions) which can predict the missing values on the test set as accurately as possible. You are allowed to use only linear regression models for this task. Cross-validation, hyperparameter tuning and regularisation are encouraged to produce better results.
  
