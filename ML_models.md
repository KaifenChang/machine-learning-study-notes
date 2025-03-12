# Machine Learning Models

## Table of Contents
1. [Random Forest](#random-forest)
2. [XGBoost](#xgboost)


***
## Random Forest
- **Definition**: 
    - An ensemble learning method 
    - Constructs multiple decision trees and combines their predictions
- **Type**: 
    - classification (Supervised learning)
    - regression (Supervised learning)
- **Key idea**: 
    - Build many decision trees using random subsets of data and features
    - aggregate their predictions to reduce overfitting and improve accuracy

### 1️⃣ Intuitive Understanding
> **Scene**: Should I bring umbrealla today?
> **Decision Tree**:  Only use a single *feature*
>  : Is it raning?
> **Random Forest**: Combination of Decision trees
> Each tree randomly choose the subset from the set of *features*
> : Is it raining?
> : Is it cloudy outside?
> : How is the humidity?

> Both tree and forest could be in single layer or multiple layers

### 2️⃣ How It Works
Step-by-step breakdown of the algorithm:
1. **Bagging (Bootstrap Aggregation)** &rarr;Random Sampling of Data
    - Random sampling of training data with replacement(會放回)
    - Construct the tree with a different bootstrap sample
    - Typically ~63% of original data used for each tree
    > e.g. 1000 datasets, but only use 700 dataset

2. **Feature Randomness** &rarr; Random Feature Selection at Each Split
    - At each split, only a random subset of features is considered
    - Classification:$\sqrt{\text{total features}}$
    - Regression: $\frac{\text{total features}}{3}$ 
    > e.g. 50 features, but randomly choose 20 features while spliting

3. **Find the Best Threshold for Splitting**
    - Each decision tree finds the best threshold (split point) independently
    - Classification: Use Gini Impurity or Entrop
    - Regression: Use Mean Squared Error
4. **Construct the decision trees**  
    - Repeat steps 1 & 2
5. **Prediction Aggregation**
    - Classification &rarr; Majority voting: (>50% threshold for binary classification)
    - Regression &rarr; Value: Average of predictions from all trees
    - Can adjust threshold for classification based on business needs (e.g., 0.7 for higher confidence)

### 3️⃣ Why Random Forest or Why not
**✅ Advantages**
- Handles missing values well.
- Works well on large datasets.
- Provides feature importance insights.

**❌ Disadvantages**
- Can be computationally expensive.
- Less interpretable than a single decision tree.
- Not always the best choice for highly imbalanced datasets.

### 4️⃣ Key Concepts
1. **Ensemble Learning**
    - Combines multiple decision trees into a "forest"
    - Each tree is trained independently
    - Final prediction aggregates results from all trees

2. **Bootstrap Sampling**
    - Random sampling with replacement
    - Each tree sees different subset of data
    - Reduces overfitting through diversity

3. **Feature Randomness**
    - Random subset of features at each split
    - Decorrelates the trees
    - Increases model robustness

4. **Voting/Averaging**
    - Classification: Majority vote from all trees
    - Regression: Average prediction from all trees
    - More reliable than single tree predictions

5. **Out-of-Bag (OOB) Error**
    - Uses unselected samples to validate each tree
    - Built-in validation mechanism
    - Helps assess model(Random Forest) performance

### 5️⃣ Scikit Learn Application
✅ **Classification:** Use `RandomForestClassifier`, **Regression:** Use `RandomForestRegressor`  
✅ **Enable `oob_score=True`** to calculate OOB error without needing an additional test set  
✅ **Use `feature_importances_`** to check feature importance  
✅ **Use `plot_tree()`** to visualize decision trees  
✅ **Use `GridSearchCV`** to tune hyperparameters such as `n_estimators`, `max_depth`, and `max_features`  
*** 

## XGBoost
- **Definition**: 
    - eXtreme Gradient Boosting 
        - A scalable and efficient implementation of gradient boosting machines
- **Type**: 
    Ensemble learning method for 
    - classification (Supervised learning)
    - regression (Supervised learning)
- **Key idea**: 
    - Builds trees sequentially
    - each tree corrects errors made by previous trees while using gradient descent optimization

### 1️⃣ Intuitive Understanding
> **Scene**: Want to boost up the grades
> e.g. Find 3 teachers
> Teacher 1: Teach you the big picture
> Teacher 2: Modify your knowing and teach you how to apply the knowkledge
> Teacher 3: Work on the application you mage and teach you how to modify the mostakes

> 1. Each teacher doesn’t just teach new material but focuses on fixing previous mistakes.
> 2. Each teacher teaches sequentially, with each one learning from the previous errors.
> 3. The final grade is a sum of all teacher efforts, not just the last teacher’s work.

### 2️⃣ How It Works
1. **Initialize and construct the first tree** 
    - Makes the initial prediction by the first tree
    - Regression: mean
    - Classification: log-odds 
2. **Calculate the Gradient and Hessian**  
   - Compute how wrong the predictions are by calculating the **gradient** (i.e., error direction).  
   - Also compute the **Hessian** (second-order derivative) to determine the step size for correction.

3. **Train a new tree to correct previous mistakes**  
   - The new decision tree learns how to correct the mistakes made by the previous tree.  
   - It does this by fitting the gradient values as new targets.
4. **Update the prediction using the new tree’s corrections**  
   - The predictions are updated by adding the new tree’s corrections
   - scaled by a learning rate \( \alpha \).  
5. **Repeat until convergence (or max trees reached)**  
   - Continue adding trees until:
     - number of trees (n_estimators) reaches the limit.
     - validation loss stops improving (Early Stopping).


### 3️⃣ Why XGBoost or Why not
**✅ Advantages**
- High performance and accuracy
- Fast training and prediction speed
- Built-in regularization to prevent overfitting
- Handles missing values automatically
- Supports parallel processing
- Memory efficient implementation
- Great for both structured and unstructured data

**❌ Disadvantages**
- More complex to tune than random forests
- Can be prone to overfitting if not configured properly
- Less interpretable than simpler models
- Requires careful parameter tuning
- May not perform well with sparse features
- Higher computational cost than basic models


### 4️⃣ Key Concepts
1. **Gradient Boosting**
    - Sequential addition of weak learners
    - Each learner focuses on previous errors
    - Gradient descent optimization

2. **Tree Structure**
    - Decision trees as base learners
    - Split points determined by gain
    - Leaf values optimized for predictions

3. **Regularization**
    - L1 and L2 penalties
    - Pruning methods
    - Learning rate control

4. **Parallel Processing**
    - Column-based (feature) parallelization
    - Cache-aware computation
    - Out-of-core computing support

5. **Hyperparameters**
    - Learning rate (eta)
    - Maximum tree depth
    - Minimum child weight
    - Subsample ratio

### 5️⃣ Scikit-Learn Application: XGBoost Best Practices
✅ Classification: Use `XGBClassifier`, Regression: Use `XGBRegressor`  
✅ Enable `early_stopping_rounds` to stop training when validation loss stops improving  
✅ Use `feature_importances_` or `plot_importance()` to check feature importance  
✅ Use `plot_tree()` to visualize individual decision trees  
✅ Use `GridSearchCV` to tune hyperparameters such as `n_estimators`, `max_depth`, and `learning_rate`  
✅ Use `colsample_bytree` and `subsample` to prevent overfitting  
✅ Use `scale_pos_weight` to handle imbalanced datasets in classification tasks  
✅ Convert data to `DMatrix` for optimized memory usage when using the native API  




***
## [Algorithm Name]
- **Definition**: 
- **Type**: 
- **Key idea**: 
### 1️⃣ Intuitive Understanding
### 2️⃣ How It Works
### 3️⃣ Why [] or Why not
**✅ Advantages**
**❌ Disadvantages**
### 4️⃣ Key Concepts
### 5️⃣ Scikit Learn Application