# MNIST_ML_Algos
This repository contains a collection of Jupyter Notebooks demonstrating different machine learning algorithms applied to the MNIST dataset. The MNIST dataset consists of handwritten digits, and the goal is to build models that can classify these digits accurately.

## Algorithms Used

- ✅ **XGBoost** (`tree_method='hist'`, parallelized with `n_jobs=-1`)
- ✅ **SVM** (Scikit-learn `SVC`, RBF kernel by default)

More models (e.g., KNN, Random Forest, or CNNs) may be added in the future.

## Observations

- 🟢 **XGBoost** trains much faster than SVM on large datasets, especially with `tree_method='hist'` and `n_jobs=-1`.
- 🔵 **SVM (RBF kernel)** achieved decent accuracy but took significantly longer to train, especially as the dataset size increased (769s to run the whole ipynb).
- 📊 Pixel normalization using `/255.0` proved more consistent and efficient than `MinMaxScaler`.
- 🧹 Preprocessing (image flattening, normalization) greatly impacts performance and training time.
