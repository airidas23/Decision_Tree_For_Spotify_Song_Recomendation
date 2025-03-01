# Spotify Music Classification with Decision Trees and Logistic Regression

## Overview
This project applies machine learning techniques to classify Spotify music data using Decision Tree classifiers and Logistic Regression. The dataset contains various audio features and a target variable indicating a classification label. The primary objectives of this project include:
- Data preprocessing and feature selection
- Training and evaluating Decision Tree models (both pruned and unpruned)
- Comparing Decision Trees with Logistic Regression
- Visualizing model decision boundaries with PCA

## Requirements
The following Python libraries are required:
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import pydotplus
from IPython.display import Image
```

## Data Preparation
1. Load the dataset:
   ```python
   data = pd.read_csv('music_spotify.csv')
   ```
2. Drop unnecessary columns:
   ```python
   data = data.drop(columns=["X", "song_title", "artist"])
   ```
3. Convert categorical variables using encoding techniques:
   ```python
   data = pd.get_dummies(data, drop_first=True)
   ```
4. Split data into features (`X`) and target (`y`):
   ```python
   X = data.drop(['target'], axis=1)
   y = data['target']
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   ```

## Model Training
1. Train a Decision Tree classifier:
   ```python
   tree_clf = DecisionTreeClassifier(min_samples_split=2, ccp_alpha=0)
   tree_clf.fit(X_train, y_train)
   ```
2. Compute the cost complexity pruning path and analyze pruning effects:
   ```python
   path = tree_clf.cost_complexity_pruning_path(X_train, y_train)
   ccp_alphas, impurities = path.ccp_alphas, path.impurities
   ```
3. Evaluate accuracy for different pruning levels:
   ```python
   ccp_alphas_collect = []
   accuracy_collect = []
   for ccp_alpha in ccp_alphas:
       tree = DecisionTreeClassifier(ccp_alpha=ccp_alpha)
       tree.fit(X_train, y_train)
       accuracy = tree.score(X_test, y_test)
       ccp_alphas_collect.append(ccp_alpha)
       accuracy_collect.append(accuracy)
   ```

## Model Comparison
We compare three models:
- **Unpruned Decision Tree**
- **Pruned Decision Tree** (with `ccp_alpha=0.01`)
- **Logistic Regression**

1. Train the pruned tree and logistic regression:
   ```python
   pruned_tree_clf = DecisionTreeClassifier(min_samples_split=2, ccp_alpha=0.01)
   pruned_tree_clf.fit(X_train, y_train)
   log_reg = LogisticRegression()
   log_reg.fit(X_train, y_train)
   ```
2. Compare model performance:
   ```python
   models = [tree_clf, pruned_tree_clf, log_reg]
   titles = ["Unpruned Decision Tree", "Pruned Decision Tree", "Logistic Regression"]
   for i, model in enumerate(models):
       train_pred = model.predict(X_train)
       test_pred = model.predict(X_test)
       train_accuracy = accuracy_score(y_train, train_pred)
       test_accuracy = accuracy_score(y_test, test_pred)
       print(f"{titles[i]} - Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")
   ```

## Results Analysis
- **Unpruned Decision Tree:** Overfits training data (Train Accuracy: ~99.9%, Test Accuracy: ~70.7%)
- **Pruned Decision Tree:** Better generalization (Train Accuracy: ~73.2%, Test Accuracy: ~69.5%)
- **Logistic Regression:** Underfits the data (Train Accuracy: ~50.9%, Test Accuracy: ~49.0%)

## Decision Boundary Visualization with PCA
To visualize decision boundaries, we apply PCA to reduce the feature space to two dimensions:
```python
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
```
Decision boundaries for each model are plotted using a custom function:
```python
def plot_decision_boundary(clf, X, y, axes, cmap=ListedColormap(['#fafab0', '#9898ff', '#a0faa0']), plot_training=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=cmap)
    if plot_training:
        plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "yo", label="Class 0")
        plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs", label="Class 1")
    plt.xlabel("$X_1$")
    plt.ylabel("$X_2$")
```
Each model’s decision boundary is plotted:
```python
plt.figure(figsize=(20, 12))
models = [tree_clf, pruned_tree_clf, log_reg]
titles = ["Unpruned Decision Tree", "Pruned Decision Tree", "Logistic Regression"]
for i, model in enumerate(models):
    plt.subplot(1, 3, i + 1)
    plot_decision_boundary(model, X_train_pca, y_train,
                           axes=[X_train_pca[:, 0].min(), X_train_pca[:, 0].max(), X_train_pca[:, 1].min(),
                                 X_train_pca[:, 1].max()])
    plt.title(titles[i])
plt.show()
```

## Conclusion
- **Decision trees provide better classification performance** compared to logistic regression for this dataset.
- **Pruning reduces overfitting** and improves generalization.
- **Visualization with PCA** gives insights into the model decision boundaries.

This project demonstrates how feature engineering, hyperparameter tuning, and visualization contribute to effective model evaluation and comparison.

