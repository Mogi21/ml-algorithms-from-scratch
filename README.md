# Machine Learning Algorithms from Scratch

This repository contains custom, ground-up Python implementations of core Machine Learning algorithms, designed to demonstrate a deep mathematical and programmatic understanding of modeling techniques.

Currently, this repository relies purely on standard Python and `numpy` for core logic, using `scikit-learn` only for dataset loading and baseline comparisons.

## 📁 Repository Contents

### 1. Ensemble Methods (`01_ensemble_methods.ipynb`)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mogi21/ml-algorithms-from-scratch/blob/main/01_ensemble_methods.ipynb)
*   **Random Forest Classifier:** Implements bootstrap sampling (bagging) and random subspace feature selection to build an uncorrelated ensemble of decision trees. Includes custom majority-voting logic.
*   **AdaBoost (Adaptive Boosting):** Implements a sequential ensemble technique that computes weighted training errors, calculates weak-learner significance ($\alpha$), and dynamically updates sample weights using exponential loss.

*Tested on the Breast Cancer dataset, achieving 95%+ accuracy.*

---
*More algorithms (Nested CV, Polynomial Lasso, SVD/Eigenfaces, K-Means, RBF Networks) will be added shortly.*# ml-algorithms-from-scratch

### 2. Unsupervised Learning & Advanced Classifiers (`02_unsupervised_and_advanced_models.ipynb`)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mogi21/ml-algorithms-from-scratch/blob/main/02_unsupervised_and_advanced_models.ipynb)
*   **SVD & Eigenfaces:** Implements Singular Value Decomposition to calculate principal components on the LFW (Labeled Faces in the Wild) dataset, alongside rank-k approximation and cumulative variance calculations.
*   **K-Means Clustering:** A custom implementation of Lloyd's algorithm for unsupervised clustering.
*   **RBF Networks:** A non-linear classifier that leverages K-Means to identify representative centers, constructs a design matrix using a Radial Basis Function kernel, and utilizes the Moore-Penrose pseudo-inverse for weight optimization. 

*Tested against the scikit-learn SVM implementation on non-linear data.*
