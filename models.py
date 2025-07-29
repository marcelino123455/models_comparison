import numpy as np
from typing import List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class LogisticRegression:
    """
    Custom Logistic Regression implementation from scratch
    """

    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000,
                 tolerance: float = 1e-6, regularization: Optional[str] = None,
                 lambda_reg: float = 0.01, random_state: Optional[int] = None):
        """
        Initialize Logistic Regression

        Args:
            learning_rate: Learning rate for gradient descent
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            regularization: Type of regularization ('l1', 'l2', None)
            lambda_reg: Regularization strength
            random_state: Random seed for reproducibility
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.random_state = random_state

        self.weights = None
        self.bias = None
        self.cost_history = []

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid activation function"""
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _add_regularization(self, cost: float) -> float:
        """Add regularization term to cost"""
        if self.regularization == 'l1':
            return cost + self.lambda_reg * np.sum(np.abs(self.weights))
        elif self.regularization == 'l2':
            return cost + self.lambda_reg * np.sum(self.weights ** 2)
        return cost

    def _regularization_gradient(self) -> np.ndarray:
        """Compute regularization gradient"""
        if self.regularization == 'l1':
            return self.lambda_reg * np.sign(self.weights)
        elif self.regularization == 'l2':
            return 2 * self.lambda_reg * self.weights
        return np.zeros_like(self.weights)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegression':
        """
        Fit the logistic regression model

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)

        Returns:
            Self for chaining
        """
        # Set random seed for reproducibility
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples, n_features = X.shape

        # Initialize parameters with Xavier/Glorot initialization
        limit = np.sqrt(6.0 / (n_features + 1))
        self.weights = np.random.uniform(-limit, limit, n_features)
        self.bias = 0.0

        # Gradient descent
        for i in range(self.max_iterations):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            predictions = self._sigmoid(z)

            # Compute cost
            cost = self._compute_cost(y, predictions)
            cost = self._add_regularization(cost)
            self.cost_history.append(cost)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            # Add regularization to weight gradients
            dw += self._regularization_gradient()

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Check for convergence
            if i > 0 and abs(self.cost_history[-1] - self.cost_history[-2]) < self.tolerance:
                print(f"Converged after {i+1} iterations")
                break

        return self

    def _compute_cost(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute logistic regression cost"""
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        cost = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return cost

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities

        Args:
            X: Feature matrix

        Returns:
            Predicted probabilities
        """
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Feature matrix
            threshold: Decision threshold

        Returns:
            Predicted class labels
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)


class NaiveBayes:
    """
    Custom Naive Bayes implementation for text classification
    """

    def __init__(self, alpha: float = 1.0):
        """
        Initialize Naive Bayes

        Args:
            alpha: Smoothing parameter (Laplace smoothing)
        """
        self.alpha = alpha
        self.class_priors = {}
        self.feature_probs = {}
        self.classes = None
        self.n_features = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'NaiveBayes':
        """
        Fit the Naive Bayes model

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)

        Returns:
            Self for chaining
        """
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        self.n_features = n_features

        # Calculate class priors
        for class_val in self.classes:
            class_count = np.sum(y == class_val)
            self.class_priors[class_val] = class_count / n_samples

        # Calculate feature probabilities for each class
        for class_val in self.classes:
            class_mask = (y == class_val)
            class_features = X[class_mask]

            # Sum of feature values for this class
            feature_sums = np.sum(class_features, axis=0)

            # Total words in this class (for normalization)
            total_words = np.sum(feature_sums)

            # Apply Laplace smoothing
            self.feature_probs[class_val] = (feature_sums + self.alpha) / (total_words + self.alpha * n_features)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities

        Args:
            X: Feature matrix

        Returns:
            Predicted probabilities for each class
        """
        n_samples = X.shape[0]
        probabilities = np.zeros((n_samples, len(self.classes)))

        for i, class_val in enumerate(self.classes):
            # Log probability to avoid numerical underflow
            class_prior = np.log(self.class_priors[class_val])

            # Sum of log probabilities for all features
            feature_log_probs = np.sum(X * np.log(self.feature_probs[class_val]), axis=1)

            probabilities[:, i] = class_prior + feature_log_probs

        # Convert back from log space and normalize
        probabilities = np.exp(probabilities)
        probabilities = probabilities / np.sum(probabilities, axis=1, keepdims=True)

        return probabilities

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Feature matrix

        Returns:
            Predicted class labels
        """
        probabilities = self.predict_proba(X)
        return self.classes[np.argmax(probabilities, axis=1)]


class SVM:
    """
    Custom Support Vector Machine implementation using gradient descent
    """

    def __init__(self, C: float = 1.0, learning_rate: float = 0.01,
                 max_iterations: int = 1000, random_state: Optional[int] = None):
        """
        Initialize SVM

        Args:
            C: Regularization parameter
            learning_rate: Learning rate for gradient descent
            max_iterations: Maximum number of iterations
            random_state: Random seed for reproducibility
        """
        self.C = C
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.random_state = random_state

        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVM':
        """
        Fit the SVM model

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,) with values -1 and 1

        Returns:
            Self for chaining
        """
        # Set random seed for reproducibility
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples, n_features = X.shape

        # Convert labels to -1 and 1 if they are 0 and 1
        y_svm = np.where(y == 0, -1, 1)

        # Initialize parameters with small random values
        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = 0.0

        # Gradient descent with simpler approach
        for epoch in range(self.max_iterations):
            for i in range(n_samples):
                # Compute margin
                margin = y_svm[i] * (np.dot(X[i], self.weights) + self.bias)

                if margin >= 1:
                    # Correct classification with sufficient margin
                    # Only regularization term
                    self.weights -= self.learning_rate * (1/self.C) * self.weights
                else:
                    # Incorrect classification or insufficient margin
                    # Hinge loss + regularization
                    self.weights -= self.learning_rate * ((1/self.C) * self.weights - y_svm[i] * X[i])
                    self.bias -= self.learning_rate * (-y_svm[i])

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Feature matrix

        Returns:
            Predicted class labels (0 or 1)
        """
        decision = np.dot(X, self.weights) + self.bias
        predictions = np.where(decision >= 0, 1, 0)
        return predictions

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Get decision function values

        Args:
            X: Feature matrix

        Returns:
            Decision function values
        """
        return np.dot(X, self.weights) + self.bias
