# Classification Algorithms

If we have a variable which takes two values (yes/no, 0/1, off/on) a linear
regression is not useful.

If we fit a line two such variable, we'll get values that make no sense.



## Classification Problem

Classification models predict a qualitative response. Their training
error is given as:

$$ \frac{1}{n} \sum_{j=1}^n I(y_i \neq \hat{y_i}) $$

Where $I()$ is the indicator function and takes value 1 if the argument
is false and 0 otherwise.

The test error rate is the average.

## Bayes Classifier

This classifier works by assigning to each observation the class $j$ for which
$\Pr(Y=j | X = x_0)$ is the largest.

So, if there are two classes 0 and 1, the label will be 0 if $\Pr(Y=0 | X = x_0) > 0.5$
and will be 1 otherwise.

This classifier is known as Bayes Classifier and its error rate is given by 

$$1 - E\left(\max_j \Pr(Y=j| X)\right) $$

## Logistic Regression

Is one of the most widely used algorithms for classification. Is also refered as
the shallow neural network.

Find a function of $y$ instead of $y$. This function (called logit) relates 
continuous values to 0 and 1. 

The logit function is defined as the log of the odds. Mathematically:

$$\log(odds) = \log\left(\frac{p}{1-p}\right) = \sum_{q=0}^Q w_qx_q = z $$

Rearranging terms:

$$\begin{aligned}
\frac{p}{1-p} &= e^z \\
\frac{1}{p} - 1 &= \frac{1}{e^z} \\
\frac{1}{p} &= \frac{1 + e^z}{e^z} \\
p &= \frac{e^z}{1+e^z} \\
p &= \frac{1}{1+e^{-z}} \\
\sigma(z) &= \frac{1}{1+e^{-z}}
\end{aligned}$$

Where $\sigma(z) = g(z)$ is the logistic function and a sigmoid.

Then the prediction is:

$$\hat{y} = \begin{cases}
0 & \text{if } \hat{p} < 0.5 \\
1 & \text{if } \hat{p} \geq 0.5 \\
\end{cases}$$

Where $\hat{p} = g(z)$.

### Decision Boundary

Once we have our weights, we can create relationships beetween
the hypothesis and the label. These relationships can be linear 
or non-linear.

- **Linear decision Boundary:**

    Supose we have a model of the form:

    $$g(z) = g(w_0 + w_1x_1 + w_2x_2)$$

    and we know that $w_0 = -5 ; w_1 = 1; w_2 = 1$.

    We can solve for $z = 0$ and create a boundary of the form $x_1 + x_2 = 5$.
    This line will separate the cases where $y=0$ and $y=1$.


- **Non-Linear decision Boundary:**

    Let's update the model to:

    $$g(z) = g(w_0 + w_1x_1 + w_2x_2 + w_3x_1^2 + w_4x_2^2)$$

    and the weights to $w_0 = -0.5 ; w_1 = w_2 = 0; w_3 = w_4 =  1$.

    If we solve for $z = 0$ we'll create a boundary of the form 
    $x_1^2 + x_2^2 = \tfrac{1}{2}$. In this case we'll have a circle that separates
    the cases where $y=0$ and $y=1$.


We could have cases where this separation becomes harder to do.

## Loss Functions for Classification

We already talked about the indicator function. So now let's talk about *margin*.

**Margin** is a measure of how correct our predictions are and it has the form 
$m = wx^Ty = zy$. If its positive then the prediction is correct.

There are times when you prefer to use margins since it doesn't depend on 
probabilities but rather on distances.

Some common margin functions are:

- **Zero-One Loss**:

    - $l_{0-1} = 1(m\leq 0)$
    - Not differentiable, doesn't account for how close you were.

- **Hinge Loss**:

    - $l_{Hinge} = \max(1-m, 0) = (1-m)_+$
    - Not differentiable at $m=1$, but
    - there's margin error when $m<1$.

- **Log Loss**:

    - $l_{Logistic} = \log(1+e^m)$
    - One of the most used since is differentiable.
    - Loss is never zero, so it always wants more margin.

Let's see how margin relates to loss functions using the *log loss*:

### Log Loss and margin

For a single training pair we know that:

$$\hat{y} = \Pr(Y=1|X=x) = \sigma(z)$$

Where $\sigma(z)$ is the logit function.

If the labels are $0$ and $1$ then $Y$ is a Bernoulli rv where:

$$\Pr(Y=y|X=x) = \hat{y}^y \cdot (1-\hat{y})^{1-y}$$

Let's take $\log$:

$$\log\left(\Pr(Y=y|X=x)\right) = y \log(\hat{y}) + (1-y)\log(1-\hat{y})$$

We now define our loss function as $L = -\log(P)$ and replace
$\hat{y} = \frac{1}{1+e^{-z}}$:

$$L = -y \log\left(\frac{1}{1+e^{-z}}\right)
-(1-y)\log\left(1-\frac{1}{1+e^{-z}}\right)$$

Let's solve both logarithms:

$$\begin{align*}
\log\left(\frac{1}{1+e^{-z}}\right) &= -\log(1) -\log(1+e^{-z}) = -\log(1+e^{-z}) \\
\log\left(1 - \frac{1}{1+e^{-z}}\right) &= \log\left(\frac{e^{-z}}{1+e^{-z}}\right) 
= -z - \log(1+e^{-z}) \\
\end{align*}$$

and replace them into the loss function:

$$ L = y\log(1+e^{-z}) +(1-y) (z + \log(1+e^{-z}))$$

Since the margin depends on signs, we'll change the mapping of $y \in \{0, 1\}$
to $y' \in \{-1, 1\}$. We do that by defining:

$$ y = \frac{1 + y'}{2} $$

We substitute into $L$ and reducing:

$$\begin{align*} L &= \frac{1 + y'}{2} \log(1+e^{-z})
+\left(1-\frac{1 + y'}{2} \right) (z + \log(1+e^{-z})) \\
&= \frac{1 + y'}{2} \log(1+e^{-z})
+\left(\frac{1 - y'}{2} \right) (z + \log(1+e^{-z})) \\
L &= \log(1+e^{-z}) +\frac{1 - y'}{2}z \\
\end{align*}$$
 
 The last step is to define margin as $m=zy'$ and analize both possible values of
 $y'$.

 - **Case 1** $y' = 1$: 

    If $y'=1\Rightarrow m=z$ and:

    $L = \log(1+e^{-m})$

 - **Case 2** $y' = -1$: 

    If $y'=-1 \Rightarrow m = -z$ and:

    $L=\log(1+e^{m}) - m$

    $L=\log(1+e^{m})-\log(e^{m})$

    $L=\log(e^{-m} + 1)$

    $L=\log(1 + e^{-m})$

So for both cases we have that $L = \log(1 + e^{-m})$. We have shown the 
relation beetween the log loss as function of the probabilities and as
function of the margin.

## Cost Function

Once we've stablished the properties of the *log loss* function let's compute
the cost function for more than one sample.

If events are independent we have:

$$ J(W) = -\log\left(\sqrt[n]{\prod_{i=1}^n \Pr(y_i | x_i)} \right) $$

Logarithm of the product is the sum of the logarithms:

$$J(W) = -\frac{1}{n} \sum_{i=1}^n y\log(\hat{y}) + (1-y)\log(1-\hat{y})$$ 

Which is known as **Binary Cross Entropy**. Is a convex function, so we can
use Gradient Descent to find the global minimum.

## Confusion Matrix

Four cuadrant matrix where we mix actual and predicted values.

<table border=1>
    <thead>
        <tr>
            <th></th>
            <th></th>           
            <th style="text-align: center;" colspan=2>Predicted</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td></td>
            <td></td>
            <td style="text-align: center; vertical-align: middle;">0</td>
            <td style="text-align: center; vertical-align: middle;">1</td>
        </tr>
        <tr>
            <td rowspan=2>Actual</td>
            <td>0</td>
            <td>True Negative</td>
            <td>False Positive</td>
        </tr>
        <tr>
            <td>1</td>
            <td>False Negative</td>
            <td>True Positive</td>
        </tr>
    </tbody>
</table>

There are some useful ratios we can compute:

- Accuracy: How many right predictions.

  $\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{Total}}$

- Recall (Sensitivity): Also known as true positive rate. Measures how many 
relevant items are identified.

  $\text{Recall} = \frac{\text{TP}}{\text{TP}+\text{FN}}$

- Precision: How many of selected items are relevant.

  $\text{Precision} = \frac{\text{TP}}{{\text{TP} + \text{FP}}}$

- $\text{F1 Score} = 2\cdot\frac{\text{Precision}\cdot\text{Recall}}{
    \text{Precision} + \text{Recall}} = \frac{\text{TP}}{
        \text{TP}+0.5\cdot(\text{FP} + \text{FN})}$

F1 is the harmonic mean beetween precision and recall.

Models can have high accuracy by correctly identifying both classes, but
that may not be what we want. That's why there's another measure called
*balanced accuracy*:

- $\text{Balanced Accuracy} = \frac{\text{sensitivity} + \text{specificity}}{2}$

We already know sensitivity. Specificity measures how many negative predictions
are correct.

  - $\text{Specificity} = \frac{\text{TN}}{\text{TN}+\text{FP}}$

To avoid problems is good to try to have similar number of samples in each
class. If not we're in front of an issue called *class imbalances*.

We can solve imbalances usins penalising algorithms, or under/over sampling (smotex).
Under/Over sampling is not useful in finance.

### ROC Curves

Other accuracy measure, called Receive Operating Characteristics curve is a 
plot of the false positive rate (1 - Specificity) versus the true positive rate.

The area under the ROC curve (AUC) is a measure of how well a model can distinguish
beetween two classes.


## Support Vector Network

- Very popular ML algorithm.
- Useful for:

    - Linear and non-linear classification
    - Regression and classification problem

- There are three kinds:

    - Hard Margin
    - Soft Margin
    - Support Vector Machine (SVM): Uses kernels

Support vector algorithms works by splitting the hyperplane in a way that maximizes
the distance beetween the groups (maximizing the margin). The vectors that
are in the margin are called support vectors and with them you can create a vector 
that is in the middle called the Decision Line.

### Mathematically

Suppose you have two groups $X_+$ and $X_-$. Also, you have a hyperplane which
is defined as $\vec{w}\cdot\vec{x} + b = 0$. Here $\vec{w}$ is a vector that 
is normal to the hyperplane (perpendicular in $\text{R}^2$). 

The hyperplane is supposed to separate the two groups so it holds:

- $\vec{w}\vec{x_+} + b > 0$
- $\vec{w}\vec{x_-} + b < 0$

Since there's no data actually passing from the hyperplane, we define two
hyperplanes that are parallel to the decision line.

- $\vec{w}\vec{x_+} + b \geq +1$
- $\vec{w}\vec{x_-} + b \leq -1$

So, the margin associated with that is:

- When the actual value $y_+ = 1 \Rightarrow y_+(\vec{w}\vec{x_+} + b) \geq +1$
- When the actual value $y_- = -1 \Rightarrow y_-(\vec{w}\vec{x_-} + b) \geq +1$

This means that the vectors that helps maximize the margin meet the condition:

- $y_i(\vec{w}\vec{x_i} + b) \geq 1$

There will be vectors that are going to be right at the parallels:

- $\vec{w}\vec{x_+} + b = +1$
- $\vec{w}\vec{x_-} + b = -1$


### Hard Margin Classifier

In the **hard margin** case, there will be samples $\vec{x_+}^*$ and $\vec{x_-}^*$
that solve the previous equations. Those are the support vectors and from them
you calculate the hyperplane. How?

The distance from the hyperplane to the support vectors is given by:

 $|\vec{w}\vec{x_i} + b| = 1$

The problem with the previous expression is that it can be changed by
the lenght of $\vec{w}$, not only $\vec{x_i}$. So, we divide by the
norm to work with a unit vector:

$\left|\frac{\vec{w}}{||\vec{w}||}\vec{x_i} + b\right| = \frac{1}{||\vec{w}||}$

So, instead of $1$, the distance is $\frac{1}{||\vec{w}||}$. This means
the distance beetween both support vectors is $\frac{2}{||\vec{w}||}$.

The maximization of the expresion found will give us the solution. But
it's easier to compute the min of $\frac{1}{2}||\vec{w}||^2$ instead. That
will give us the maximal margin.

To recap, the **hard margin classifier** is the result off:

$$\begin{aligned}
& \arg\min \frac{1}{2}||\vec{w}||^2 \\
& \text{subject to} \\
& y_i (\vec{w}\vec{x_i} + b) - 1 = 0
\end{aligned}$$

### Support Vector Classifier (SVC) - Soft Margin

The hard margin is theoretical mostly. It's not easy to find something
like that. We need to relax the constraint. That's why we have the 
**soft margin classifier** also known as the **Support Vector Classifier**:


$$\begin{aligned}
& \arg\min \frac{1}{2}||\vec{w}||^2 + C\sum_i^n \zeta_i\\
& \text{subject to} \\
& y_i (\vec{w}\vec{x_i} + b) \geq (1 - \zeta_i) 
\end{aligned}$$

Here the $\zeta_i$ is the *relaxation* of the model. If:

-  $\zeta_i = 0$: The vector does not reside inside the margin (like the initial
example)
-  $0 < \zeta_i < 1$: The vector resides inside the margin
-  $\zeta_i = 1$: The vector resides on the hyperplane
-  $\zeta_i > 1$: The vector is in the wrong side of the hyperplane.

So there's a trade of between the first part of the minimization and the 
second one. You can have a smaller $||\vec{w}||$ by allowing the vectors
to lie inside the margins, but that increases the penalization.

$C$ is a non-negative tuning parameter. Bigger values of $C$ means the
penalization for being inside the margins is bigger.


How to define the values of $\zeta_i$? It will be the distance
of the the $x_i$ that's beyond its margin to the margin. If the
distance is greater than one, then the vector is already in its group
and as such there's no need for penalization. So the definition will be:

$$\zeta_i = \max(0, 1-y_i(\vec{w}\vec{x_i} + b))$$

Note that the expression can be written as:

$$\zeta_i = \max(0, 1-m))$$

Which is the *hinge loss formula*. If we rewrite the minimization problem we have:

$$\arg\min \frac{1}{2}||\vec{w}||^2 + C\sum_i^n \max(0, 1-m)$$

### Support Vector Machines (SVM)

Hard and soft margins work when the data is linearly separable. When that's 
not the case, we use the *kernel trick* to calculate distances in higher dimensions
where the data is linearly separable.

How does this work? Let's start by solving the optimization problem:

#### Lagrangian Dual Problem
We start by creating the lagrangian problem:

$$\begin{flalign}
&&\\
\arg\min_{\vec{w}, b} L 
= \frac{1}{2}||\vec{w}||^2-\sum_i^n \alpha_i (y_i (\vec{w}\vec{x_i} + b) - 1)
\end{flalign}$$

Differentiate the vectors:

$$\begin{flalign}
&&\\
\frac{\partial ||\vec{w}||^2}{\partial \vec{w}} = 2\vec{w} \qquad
\text{and} \qquad \frac{\partial \vec{x}\vec{w}}{\partial{w}} = \vec{x}
\end{flalign}$$

Find the derivatives of the Lagrangian $L$:

$$
\begin{flalign}
&&\\
& \frac{\partial L}{\partial \vec{w}} = \vec{w} - \sum_i^n \alpha_iy_i\vec{x_i} = 0 
\Rightarrow \vec{w} = \sum_i^n \alpha_iy_i\vec{x_i} \\
& \\ 
& \frac{\partial L}{\partial b} = \sum_i^n \alpha_iy_i = 0 
\end{flalign}
$$

Let's solve $L$:


$$\begin{flalign}
&&\\
L &= \frac{1}{2}\left(\sum_i^n \alpha_iy_i\vec{x_i}\right)\left(\sum_j^n \alpha_jy_j\vec{x_j}\right)
-\sum_i^n \alpha_i \left(y_i \left(\sum_j^n \alpha_jy_j\vec{x_j} \vec{x_i} + b\right) - 1\right) \\
&= \frac{1}{2}\left(\sum_i^n \alpha_iy_i\vec{x_i}\right)\left(\sum_j^n \alpha_jy_j\vec{x_j}\right)
-\sum_i^n \alpha_i \left(y_i\sum_j^n \alpha_jy_j\vec{x_j} \vec{x_i} + y_ib - 1\right) \\
&= \frac{1}{2}\left(\sum_i^n \alpha_iy_i\vec{x_i}\right)\left(\sum_j^n \alpha_jy_j\vec{x_j}\right)
-\sum_i^n \alpha_iy_i\sum_j^n \alpha_jy_j\vec{x_j} \vec{x_i} + b\cancelto{0}{\sum_i^n \alpha_iy_i} + \sum_i^n \alpha_i \\
&= \frac{1}{2}\left(\sum_i^n \alpha_iy_i\vec{x_i}\right)\left(\sum_j^n \alpha_jy_j\vec{x_j}\right)
-\sum_i^n \alpha_iy_i\vec{x_i} \sum_j^n \alpha_jy_j\vec{x_j} + \sum_i^n \alpha_i \\
L &= \sum_i^n \alpha_i - \frac{1}{2}\left(\sum_i^n \alpha_iy_i\vec{x_i}\right)
\left(\sum_j^n \alpha_jy_j\vec{x_j}\right) \\
L &= \sum_i^n \alpha_i - \frac{1}{2}\sum_{i,j}^n \alpha_iy_i\alpha_jy_j\vec{x_i}\vec{x_j}
\end{flalign}$$

From the expression above we can see that the minimization problem depends on the product beetween the 
vectors. When the problem is linearly separable this works, because it can be shown that the classification
problem will depend solely on this product. But if the problem is not linearly separable the product
will not tell us that because the direction will not be relevant in this dimension.

To solve the problem we use the kernel trick. If we add dimensions to the problem we can find a place
where the problem is linearly separable. To do this we apply a transformation $\Phi(\vec{x})$ so that
$\Phi(\vec{x_i}) \cdot \Phi(\vec{x_j})$ will be meaningful in telling us to which class the vector
belongs.

The problem is that finding and working with $\Phi$ is quite hard. That's where we use the *kernel trick*
which tells us that we can find a function $k(\vec{x_i}, \vec{x_j})$ called *kernel* which is equivalent to 
$\Phi(\vec{x_i}) \cdot \Phi(\vec{x_j})$ and uses the vectors withouth the change of dimension.

The above problem is then updated to:

$$\begin{flalign}
&&\\
L &= \sum_i^n \alpha_i - \frac{1}{2}\sum_{i,j}^n \alpha_i\alpha_j\cdot y_iy_j\cdot k(\vec{x_i},\vec{x_j})
\end{flalign}$$

### Kernels

Kernels are hard to find, here are some useful ones. We must choose the one that works best
for our hyperplane:

- Linear: $(\vec{x_i}, \vec{x_j})$
- Polynomial: $(r + \vec{x_i}\cdot\vec{x_j})^d$
- Gaussian RBF: $\exp(-\gamma||\vec{x_i}-\vec{x_j}||^2)$ or $\exp\left(-\frac{x - z}{2\sigma^2}\right)$
- Sigmoid: $\tanh(\gamma\vec{x_i}\cdot\vec{x_j} + r)$

### SVM for Regression

Here you want the opposite, i.e., you want vectors to lie inside the margin. So the restriction
have the opposite sign. We update the SVC to this case:

$$\begin{aligned}
&\arg\min \frac{1}{2}\|\vec{w}\|^2 + C\sum_{i}^{n} (\zeta_{i} + \zeta_{i}^*)\\
&\text{subject to} \\
&y_{i} -\vec{w}\vec{x_{i}} - b \leq (\epsilon + \zeta_{i}) \\
&\vec{w}\vec{x_{i}} + b - y_{i} \leq (\epsilon + \zeta_{i}^*) 
\end{aligned}$$

The cost function is called well-loss

## KNN

- One of the simplest ML algorithm. Called a *lazy algorithm* as it doesn't technically train a model.
- In K-Nearest Neighbor we set an integer *K* as the number of neighbors we want to look.
- The algorithm works as it follows:
    1. Compute the distance
    1. Sort by ascending distance to find the *k* NN.
    1. Compute the average or probability of *k* NN being of a label. 


## Hyper-parameter Tuning

Hyper-parameters are those that are not directly learnt within estimators. In the SVC, for example,
$C$ is a hyper-paramether.

The hyper-parameter tuning is a process to choose the best values for these parameters in order
to improve the cross validation score.

Some approaches to hyper parameter tuning are:

- **Grid Search**: Convetional way, where we look for a manually specified subset of the 
hyper-parameter space. We perform an exhaustive search.
- **Random Search**: Here there's no manually specified subset, is random which makes
this model very noisy
