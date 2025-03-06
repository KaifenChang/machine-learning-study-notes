# General Python Coding and Machine Learning Notes
### Table of Contents
1. [Docstrings](#docstrings)
2. [Parameter vs Argument](#parameter-vs-argument)
3. [Matplotlib vs Seaborn](#matplotlib-vs-seaborn)
4. [Handling Imbalanced Data](#handling-imbalanced-data)
    - [SMOTE](#smote)
    - [Random Undersampling](#random-undersampling)
5. [Pipelines](#pipelines)
6. [Train and Test Split](#train-and-test-split)
7. [Dot notation and function()](#dot-notation-and-function)
8. [Built-in functions](#built-in-functions)
    - [hasattr()](#hasattr)
9. [Classification Report](#classification-report)
10. [Confusion Matrix](#confusion-matrix)
11. [PR and ROC Curve](#pr-and-roc-curve)
    - [Precision-Recall Curve](#precision-recall-curve)
    - [ROC Curve](#roc-curve)

*** 

## Docstrings
``` python
def function(parameter1, parameter2):
    """
    What is this function doing

    Args:
        parameter1 (type): doing...
        parameter2 (type): doing...

    Returns:
        return_type: description of what is returned
    """
    # Function code goes here
```

e.g. 
```python
def handle_imbalanced_data(X_train, y_train, sampling_strategy=0.1):

"""

Handle imbalanced data using SMOTE and undersampling.

Args:

X_train (pd.DataFrame): Training features

y_train (pd.Series): Training target

sampling_strategy (float): Ratio of minority to majority class after resampling

Returns:

tuple: Resampled X_train, y_train

"""
```

*** 


## Parameter vs Argument


**Parameter（參數）**:
- from function definition
- like a placeholder

**Argument（引數）**:
- from function call
- like a value
- used in docstring to 
    - describe the expected arguments rather than the internal parameter names

*** 
## Matplotlib vs Seaborn
**Matplotlib**:
- low-level
- more flexible
- more customizable

**Seaborn**:
- high-level
- statistical data visualization library built on top of Matplotlib
    - e.g.
        - `sns.barplot()`
        - `sns.lineplot()`
        - `sns.scatterplot()`
        - `sns.heatmap()`
        - `sns.pairplot()`
        - `sns.heatmap()`

*** 

## Handling Imbalanced Data

### SMOTE
Synthetic Minority Over-sampling Technique （合成少數過採樣技術）
**For generating synthetic samples for minority class.**
| Step | Description | Details |
|------|-------------|---------|
| **Step 1** | Identify the minority class | Determine which class has fewer samples |
| **Step 2** | Find k-nearest neighbors (KNN) | Choose k-nearest neighbors (commonly $k = 5$) based on Euclidean distance |
| **Step 3** | Generate synthetic samples | 1. Randomly select a minority class sample $x$<br>2. Choose a random nearest neighbor $x_2$<br>3. Compute difference: $diff = x_2 - x$<br>4. Generate new sample: $x_{new} = x + diff \times \lambda$ where $\lambda \sim U(0,1)$ |
| **Step 4** | Repeat until dataset is balanced | Continue Steps 2-3 until the desired number of synthetic samples is created |

**Example**:
For a minority class sample $x$:
1. Find 5 nearest neighbors using KNN: $x_1, x_2, x_3, x_4, x_5$
2. Randomly select one neighbor, e.g. $x_2$
3. Calculate difference: $diff = x - x_2$
4. Generate new sample: $x_{new} = x_2 + diff \times random(0,1)$

The synthetic sample $x_{new}$ will be located between $x$ and $x_2$.

**Why SMOTE is effective?**
- suitable for supervised learning
- Prevent **overfitting** by generating synthetic samples than duplicate minority samples(random oversampling)

**When to use SMOTE?**
- Dataset is **highly imbalanced** (minority class <10% of total data)
- The dataset has **numerical** features (SMOTE is not ideal for categorical data).
- You are using models sensitive to imbalance (e.g., Decision Trees, SVMs, Neural Networks).
- The dataset has a clear structure and minimal noise.

**When to avoid SMOTE?**
- If the dataset is in **high-dimensional**
- if the dataset has **categorical** features

**How to implement SMOTE in Python?**
```python
from imblearn.over_sampling import SMOTE

smote_dataset = SMOTE(
    sampling_strategy='auto', # auto: default, minority = majority
    random_state=42,
    k_neighbors=5
)

X_res, y_res = smote_dataset.fit_resample(X, y)
```

### Random Undersampling
For **decreasing** the number of majority class samples.

| Step | Description |
|------|-------------|
| **Step 1** | Identify the majority and minority class |
| **Step 2** | Decide the target number of majority class samples |
| **Step 3** | Randomly select and remove the remaining samples |
| **Step 4** | Ensure the dataset is balanced |

**Why Random Undersampling?**
- For highly imbalanced dataset
- Prevent **overfitting** by reducing the number of majority samples

**Are there any disadvantages?**
- Loss of information from majority class
- Potential to discard useful information

**How to implement Random Undersampling in Python?**
```python
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(
    sampling_strategy='auto',
    random_state=42
)

X_res, y_res = rus.fit_resample(X, y)
```

*** 

## Pipelines
1. To run two or more actions in a sequence
2. Use `[('step1', obj1), ('step2', obj2)] ` to specify the sequence of actions

**Why use pipelines?**
- For cleaner code and better readability.
- To avoid data leakage.
- hyperparameter tuning (`GridSearchCV`)
- can be used as an estimator in scikit-learn

**How to use pipelines?**
```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('step1', obj1),
    ('step2', obj2),
    ...
])
```

***
## Train and Test Split
X_train: input data for training
X_test: input data for testing
y_train: target data for training
y_test: target data for testing

**Why split the data?**
- To evaluate the performance of the model
- To avoid overfitting


***
## Dot notation and function()

**Dot notation**
- `module.function()` e.g. `pd.read_csv()` 👉 「從模組拿工具來用」
- `obj.method()` e.g. `df.describe()` 👉 「讓物件執行某個方法」
- `obj.attribute` e.g. `df.shape` 👉 「取得物件的某個屬性值」
- `module.attribute` e.g. `np.pi` 👉 「模組內定義的變數或常數」

**Function**
- need additional arguments for the specific output: `function(obj, arg1, arg2)`
- don't need additional arguments and will return a fixed output: `function(obj)`

***
## Built-in functions

### `hasattr()`

```python
hasattr(obj, 'method')
```
check if an object has a specific attribute or method (Bool)

***
## Classfication Report

| Metric | Description | Formula | Notes |
|--------|-------------|---------|--------|
| **Precision** (精確率) | 模型預測為「正類別」時，有多少是正確的 | TP / (TP + FP) | - |
| **Recall** (召回率) | 實際為「正類別」時，模型預測為「正類別」的比例 | TP / (TP + FN) | 高：模型預測的正確率高<br>低：模型預測的正確率低 |
| **F1 Score** (F1值) | 精確率和召回率的調和平均 | 2 * (Precision * Recall) / (Precision + Recall) | 越趨近於1 越好 |
| **Support** | Number of actual occurrences of the class in the dataset | - | - |

- accuracy: 模型預測正確的數量 / 總數
- macro avg: 各類別的Precision、Recall、F1 Score的平均值
- weighted avg: 各類別的Precision、Recall、F1 Score的加權平均值

***
## Confusion Matrix
`cm = confusion_matrix(y_test, y_pred)`

| Actual vs Predicted | Predicted 0 | Predicted 1 |
|-------------------|-------------|-------------|
| Actual 0 (Negative) | TN (True Negative 真負例) | FP (False Positive 假正例) |
| Actual 1 (Positive) | FN (False Negative 假負例) | TP (True Positive 真正例) |


***
## PR and ROC Curve

### Precision-Recall Curve
For imbalanced dataset, PR curve is more informative than ROC curve.
- x-axis: recall
- y-axis: precision
- Threshold: 0.5
    - if y_prob > 0.5, then it is a positive sample
    - if y_prob <= 0.5, then it is a negative sample

    | Threshold Level | Value | Effect | Example Use Case |
    |----------------|--------|---------|-----------------|
    | Very High | 0.9 | Most samples predicted as positive | Medical diagnosis |
    | High | 0.5 | More samples predicted as positive | General cases |
    | Low | 0.3 | Less samples predicted as positive | Fraud detection |
    | Very Low | 0.1 | Few samples predicted as positive | Medical screening |

**PR curve:**
- good model: high precision and high recall
    - go to right and up
- if the curve decrease: can handle both precision and recall
- PR AUC (area under curve): bigger is better


### ROC Curve
Receiver Operating Characteristic Curve
- For the overall performance of the model
- Analyzing the effect of threshold on the model:
    - x-axis: FPR (False Positive Rate)
    - y-axis: TPR (True Positive Rate)



**ROC curve:**
| Model Type | AUC Range | Curve Characteristic |
|------------|-----------|---------------------|
| Perfect | AUC = 1 | Close to the top left corner |
| Excellent | AUC = 0.9-1 | Close to the top left corner |
| Good | AUC = 0.7-0.9 | Close to the top left corner |
| Bad | AUC = 0.5-0.7 | Close to the bottom right corner |
| Random | AUC = 0.5 | Diagonal line |
| Poor | AUC < 0.5 | Below the diagonal line |


### PR vs ROC
- PR curve is more informative than ROC curve for imbalanced dataset
- ROC curve is more informative than PR curve for balanced dataset

| Aspect | PR Curve | ROC Curve |
|--------|----------|------------|
| X-axis | Recall | False Positive Rate (FPR) |
| Y-axis | Precision | True Positive Rate (TPR) |
| Best Point | Top-right corner (1,1) | Top-left corner (0,1) |
| Dataset Type | Better for imbalanced datasets | Better for balanced datasets |
| Sensitivity to Class Imbalance | Less sensitive | More sensitive |
| Focus | Focuses on **positive class** performance | Focuses on both class performances |
| Use Case | When positive class is rare/important | When both classes are equally important |
| Baseline | Changes with class distribution | Always a diagonal line (AUC=0.5) |




