# Supervised Learning I


## Linear Regression

Predicting response $y$ from a set of predictor variables $X$.

$$y = w\cdot X$$

Vector $w$ is unknown. Goal is to achieve this such as to minimize the
error beetween actual and estimated values.

## Loss functions for LR

There are three known loss functions all of which are translation-invariant,
i.e. they don't capture relative error:

- L1 or Laplacian: Absolute value of the loss
- L2 or quadratic error: Squared loss 
- Huber: It's behaviour depends on a threshold level $\delta$:

    $L_\delta(r) = 
        \begin{cases}  
        \frac{1}{2} r^2 & \text{for } |r| \leq \delta \\
        \delta(|r| - \frac{1}{2}\delta) & \text{otherwise}
        \end{cases}
        $

## Gradient Descent

- Optimisation algorithm used to find optimal solutions.
- Use the function's gradient to move step by step from an initial solution to
the optimal one.
- Gradient Descent doesn't differentiate beetween local and global minimum.
- Steps are given as:

    $w_{new} = w_{old} - \eta \frac{\partial J(W)}{\partial W}$ 

    Where $\eta$ is the learning rate.

### Batch gradient descent

GD starts from a single initial point, which means its solution is a noisy and
not very reliable one. We can instead take a batch of points, compute the 
solution and then take the mean of them. That's called *batch gradient descent*
and comes in three flavours:

- Batch gradient descent:

    The batch size is equal to the size of the training set

- Stochastic Gradient Descent:

    The batch size is equal to 1.

- Mini-batch Gradient Descent:

    Anything in between Batch and Stochastic.

## Learning rate

If $\eta$ is very small, computations will take too much time. If it's large,
then you risk never converging. So we need to find the appropiate learning
rate.

How we do it?

We can select fixed values and try or we can use adaptative learning rates.
Adaptative uses functions to automatically change learning rates.

## Overfitting

We address overfitting with regularisation methods, such as:

1. Dropout:

    Used in NN by dropping neurons.

1. Early stopping:

    Stop the training as soon as the validation error starts to
    increase again, based on an parameter called patience.

1. Other methods:

    1. Manual selection of features
    1. Model selection algorithms:

        - Resampling methods: Cross Validation.
        - Subset selection: Feature selection.
        - Shrinkage methods: Penalty regressions such as Lasso and Ridge.

Basically, with regularisation we try to reduce manitude of coefficients.

### Resampling Methods

- We draw samples repeatedly from a training set and refit a model on each sample.
This methods can be computationally expensive, something to keep in mind.

- The most commonly used is the cross-validation.

    - Split the training set in various folds and uses each time a 
    different fold as a test data.
    - This is for finding the parameters, not for testing the whole model.
    - Not useful for time series since it mixes future and past data.

- Time Series Cross Validation (TSCV):

    These methods solve the issue with x-validation:
    - Sliding window: Same size training/test batches moving from start to finish
    - Forward Chaining: Fixed size test batch, training set starts the same size
    but keeps increasing as we move into the future.

### Subset selection methods

- Reduce the number of input variables
- Three categories:

    - Filter: Generic, not incorporated into the algorithm. Ex: Checking the
    correlation beetween the variables.
    - Wrapper: Specific ML algorithm to find optimal features, like Forward
    selection method or Shap.
    - Embedded: Feature selection is done in the training phase by the model.
    Ex: Lasso or ElasticNet.

- There are other unsupervised methods to do this.
- Always try to use more than one method.

### Shrinkage Methods

Methods that add a penalization part into the loss function.

- **Lasso**:

    - Least Absolute Shrinkage and Selection Operator
    - Adds a L1 penalty so the cost function looks like:

      $L = MSE(RSS) + \lambda \sum_{j=1}^p |w_j| $

      Where $\lambda$ is the regularisation penalty.
    - Shrinks some of the coefficients to zero which is useful for
    feature selection.

- **Ridge**:

    - Adds a L2 penalty so the cost function looks like:

      $L = MSE(RSS) + \lambda \sum_{j=1}^p w_j^2 $

      Where $\lambda$ is the regularisation penalty.

    - Shrinks coefficients in a smoother way.

- **ElasticNet**:

    - It's a combination of the previous methods.
    - The loss function is:

      $L = MSE(RSS) + \lambda \left(
        \frac{1-\alpha}{2} \sum_{j=1}^p w_j^2 + \alpha \sum_{j=1}^p |w_j|\right) $

      Where $\lambda$ is the regularisation penalty and $0\leq\alpha\leq 1$.

    - *Don't try to do $\alpha$ very close to zero do to destabilization issues.*