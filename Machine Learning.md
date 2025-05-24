# Supervised Learning

## Model Development Steps

1. **Understand model:**
    
    Write your model, what you want to do, with which data, timeframes, instruments and so on.

1.  **Collect data:**

    Using whether a private or public API

1. **Explore / Visualization**:

    - Exploratory data analysis (EDA)
    - Expand features:

        From one price series you can generate several technical indicator, for example.

    - Visualize:
        * Easier to find splits, coupons, missing values and so on.
        * You can see differences beetween the features you're creating.

1. **Clean**
1. **Transform**

    - Imputation
    - Interpolation
    - *If more than 5% of data is missing you probably would be better off dropping or finding better data.*
    - Transformation:
        * Switch scaling:

            Important because many models are based on distances. Using **EDA** you can choose the best approach
            to scaling and is very critical to the process.
            - minmax
            - normalization
            - standardization
            - robust scaling (useful for commodities and crypto)


1. [**Model**](#predictive-modeling-process)

    Up until now, all the work has been on the data. That's the most important part of the job. Models are
    much easier to try and change and it's important to understand that *garbage in, garbage out.*

    - Linear regressions, logits
    - XB
    - NN
    - DNN

1. **Validation**

    A two step:
        1. Backtesting: You see how the signals work on the test sample.
        1. Metrics: MSC, RMSC (for regression), Binary cross entropy, log loss (for classification)

1. **Deployment**

    You move your model into production


All the previous  steps are iterative. It usually takes 6 months to develop a strategy.


## Predictive Modeling Process

1. Preprocessing
    - Define labels.
    - Split training and test dataset.

1. Learning
    - Model selection
    - Cross-validation (Time series cross validation in case of financial time series)
    - Performance metrics (like the ones described above)
    - Hyperparameter optimization:

        Run the model, change parameters and then compare with your initial baseline model.

1. Evaluation:
    - Run backtests on data
    - Run the model on the test dataset.

## Feature Engineering

Use domain knowledge to select and transform the most relevant
features or variables from raw data.


## Bias - Variance Tradeoff

Bias is how far your expected value is from the actual value. Variance is how
sensitive the model is to changes in the data.

If you want to achieve 0 bias, you will need to overfit the model thus creating
high variance. If you want 0 variance you will need a very unelastic model (think 
about a model that always assign the same value) which will have a huge bias. 

In other words, you need to make a trade off beetween accuracy and overfitting.