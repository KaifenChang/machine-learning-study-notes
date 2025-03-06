# General Python Coding and Machine Learning Notes
### Table of Contents
1. [Docstrings](#docstrings)
2. [Parameter vs Argument](#parameter-vs-argument)
3. [Matplotlib vs Seaborn](#matplotlib-vs-seaborn)
4. [SMOTE](#smote)
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

## Parameter vs Argument


**Parameter（參數）**:
- from function definition
- like a placeholder

**Argument（引數）**:
- from function call
- like a value
- used in docstring to 
    - describe the expected arguments rather than the internal parameter names


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

## SMOTE