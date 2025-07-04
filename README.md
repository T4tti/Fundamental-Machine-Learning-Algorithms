# Fundamental Machine Learning Algorithms - Complete Documentation

## Overview

This repository contains comprehensive implementations of fundamental machine learning algorithms and Python utilities. It serves as a complete resource for learning and understanding core ML concepts with practical implementations.

## Repository Structure

```
fundamental-algorithm-machine-learning/
├── README.md
├── K-Nearest_Neighbors/
│   ├── K-NN_Classification.py
│   └── K-NN_Regression.py
├── Linear_Regression/
│   └── Linear_Regression.py
├── Logistic_Regression/
│   └── Logistic_Regression.py
├── Naive_Bayes/
│   ├── Gaussian_NB.py
│   └── Multinomial_NB.py
└── Tutorial_Python/
    └── List/
        ├── add_noise_augmentation.py
        ├── calcu_accuracy.py
        ├── create_bow_vector.py
        ├── filter_box.py
        ├── flatten_tokens.py
        ├── min_max_scale.py
        ├── one_hot_encode.py
        ├── pad_sequences.py
        └── split_dataset.py
```

## Machine Learning Algorithms

### 1. K-Nearest Neighbors (K-NN)

#### Classification (K-NN_Classification.py)

- **Purpose**: Classification using K-nearest neighbors algorithm
- **Features**:
  - Configurable distance metrics (Minkowski distance with parameter p)
  - Manhattan distance (p=1) and Euclidean distance (p=2) support
  - 3D visualization with matplotlib
  - Interactive plotting with connecting lines to neighbors

- **Key Components**:
  - `minkowski_distance`: Computes Lp norm distance
  - `KNN` class with `fit()` and `predict()` methods
  - 3D scatter plot visualization with decision boundary

#### Regression (K-NN_Regression.py)

- **Purpose**: Regression using K-nearest neighbors algorithm
- **Features**:
  - Predicts continuous values by averaging k nearest neighbors
  - Configurable distance metrics
  - 2D visualization for regression problems

- **Key Components**:
  - `minkowski_distance`: Distance calculation function
  - `KNN` class with regression capabilities

### 2. Linear Regression (Linear_Regression.py)

- **Purpose**: Linear regression with gradient descent optimization
- **Features**:
  - Gradient descent implementation with convergence checking
  - Real-time visualization of learning process
  - Performance metrics (MSE, MAE)
  - Animated regression line updates during training

- **Key Components**:
  - `LinearRegression` class
  - `MSE` and `MAE` evaluation metrics
  - `gradient_descent` optimization method

### 3. Logistic Regression (Logistic_Regression.py)

- **Purpose**: Binary classification using logistic regression
- **Features**:
  - Two optimization approaches: gradient descent (log-loss) and gradient ascent (log-likelihood)
  - Sigmoid activation function
  - Real-time visualization of decision boundary evolution
  - Probability predictions

- **Key Components**:
  - `sigmoid`: Activation function
  - `LogisticRegression` class with dual optimization methods
  - Interactive plotting of training progress

### 4. Naive Bayes

#### Gaussian Naive Bayes (Gaussian_NB.py)

- **Purpose**: Classification assuming Gaussian distribution of features
- **Features**:
  - Gaussian probability density function implementation
  - Prior and posterior probability calculations
  - Iris dataset example with visualization

- **Key Components**:
  - `Gaussian_NB` class
  - `_pdf`: Probability density function
  - Scatter plot visualization comparing actual vs predicted

#### Multinomial Naive Bayes (Multinomial_NB.py)

- **Purpose**: Classification for discrete features (e.g., text classification)
- **Features**:
  - Laplace smoothing (alpha parameter)
  - Log probability calculations for numerical stability
  - Sklearn-compatible interface

- **Key Components**:
  - `Multinomial_NB` class
  - `predict_log_proba`: Log probability predictions
  - Feature log probability calculations

## Python Utilities (Tutorial_Python/List/)

### Data Preprocessing

#### 1. Sequence Padding (pad_sequences.py)
- **Purpose**: Standardize sequence lengths by padding with zeros
- **Function**: `pad_sequences`
- **Use Case**: NLP preprocessing for uniform input lengths

#### 2. Data Splitting (split_dataset.py)
- **Purpose**: Split datasets into train/validation/test sets
- **Function**: `split_dataset`
- **Features**: Configurable ratios with comprehensive test cases

#### 3. Feature Scaling (min_max_scale.py)
- **Purpose**: Normalize features to [0,1] range using Min-Max scaling
- **Function**: `min_max_scale`
- **Features**: Handles edge cases (constant values, empty data)

### Text Processing

#### 4. Token Flattening (flatten_tokens.py)
- **Purpose**: Flatten nested token lists for vocabulary creation
- **Function**: `flatten_tokens`
- **Use Case**: NLP preprocessing for corpus analysis

#### 5. One-Hot Encoding (one_hot_encode.py)
- **Purpose**: Convert categorical labels to one-hot vectors
- **Function**: `one_hot_encode`
- **Features**: Handles arbitrary class orders and label types

#### 6. Bag-of-Words (create_bow_vector.py)
- **Purpose**: Convert documents to Bag-of-Words representation
- **Function**: `create_bow_vectors`
- **Use Case**: Text classification feature extraction

### Computer Vision

#### 7. Bounding Box Filtering (filter_box.py)
- **Purpose**: Filter object detection predictions by confidence threshold
- **Function**: `filter_low_confidence_boxes`
- **Format**: `[class_id, confidence, x, y, w, h]`

### Data Augmentation

#### 8. Noise Augmentation (add_noise_augmentation.py)
- **Purpose**: Add Gaussian noise to time series for data augmentation
- **Function**: `add_noise_augmentation`
- **Features**: Configurable noise levels using Gaussian distribution

### Evaluation

#### 9. Accuracy Calculation (calcu_accuracy.py)
- **Purpose**: Calculate classification accuracy
- **Function**: `calculate_accuracy`
- **Features**: Handles various data types and edge cases

## Key Features Across All Implementations

### Visualization
- **3D plotting**: K-NN classification with interactive visualization
- **Real-time training**: Linear and logistic regression with animated learning
- **Comparison plots**: Actual vs predicted results for all classifiers

### Robustness
- **Error handling**: Input validation and edge case management
- **Convergence checking**: Early stopping for optimization algorithms
- **Comprehensive testing**: Extensive test cases for utility functions

### Educational Value
- **Clear documentation**: Detailed comments explaining algorithms
- **Multiple approaches**: Different optimization methods (gradient descent/ascent)
- **Practical examples**: Real datasets (Iris) and synthetic data demonstrations

## Usage Examples

### Machine Learning Models
```python
# K-NN Classification
clf = KNN(k=3, p=2)
clf.fit(points)
prediction = clf.predict(new_point)

# Linear Regression
model = LinearRegression(learning_rate=0.001, n_iterations=1000)
model.fit(X, y)
predictions = model.predict(X_test)

# Logistic Regression
model = LogisticRegression(learning_rate=0.1, n_iterations=600)
model.fit(X, y, method="gradient_descent_log_loss")
```

### Utility Functions
```python
# Data preprocessing
padded_sequences = pad_sequences(sequences)
train, val, test = split_dataset(data, 0.7, 0.15)
scaled_data = min_max_scale(features)

# Text processing
flattened = flatten_tokens(corpus)
one_hot = one_hot_encode(labels, classes)
bow_vectors = create_bow_vectors(corpus, vocabulary)
```

## Dependencies
- `numpy`: Numerical computations
- `matplotlib`: Visualization
- `pandas`: Data manipulation (Logistic Regression)
- `sklearn`: Dataset loading and train/test splitting
- `collections`: Counter for K-NN classification
- `random`: Gaussian noise generation

## Getting Started

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd fundamental-algorithm-machine-learning
   ```

2. **Install dependencies**:
   ```bash
   pip install numpy matplotlib pandas scikit-learn
   ```

3. **Run examples**:
   ```bash
   # K-NN Classification
   python K-Nearest_Neighbors/K-NN_Classification.py
   
   # Linear Regression
   python Linear_Regression/Linear_Regression.py
   
   # Logistic Regression
   python Logistic_Regression/Logistic_Regression.py
   
   # Naive Bayes
   python Naive_Bayes/Gaussian_NB.py
   python Naive_Bayes/Multinomial_NB.py
   ```

This repository provides a comprehensive foundation for understanding and implementing fundamental machine learning algorithms with practical Python utilities for data processing and analysis.
