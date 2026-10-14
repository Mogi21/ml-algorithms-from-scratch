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
