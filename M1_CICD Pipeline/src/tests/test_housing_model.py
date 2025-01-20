# tests/test_housing_model.py

import numpy as np
import pytest
from sklearn.model_selection import train_test_split
from src.housing_model import HousingModel

def create_sample_data(n_samples=100):
    """
    Creates sample data for testing the HousingModel.
    :param n_samples: Number of samples to generate.
    :return: Tuple of feature matrix X and target vector y.
    """
    np.random.seed(42)  # For reproducibility
    X = np.random.rand(n_samples, 2)  # 2 feature variables
    y = 3 * X[:, 0] + 5 * X[:, 1] + np.random.normal(0, 0.1, n_samples)  # Target variable
    return X, y

def test_housing_model():
    # Generate sample data
    X, y = create_sample_data()
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create an instance of HousingModel
    model = HousingModel()
    
    # Train the model
    model.train(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Verify the predictions are of the correct shape
    assert predictions.shape == y_test.shape, "Predictions shape does not match target shape."

    # Further check that the model is indeed making predictions (that they are within a reasonable range)
    assert np.all(np.isfinite(predictions)), "Predictions contain non-finite values."
    assert np.all(predictions >= 0), "Predictions should be non-negative."  # Assuming housing prices cannot be negative.

def test_model_training():
    # Test the training process on a small dataset
    X, y = create_sample_data(n_samples=10)
    model = HousingModel()

    # Train the model
    model.train(X, y)

    # Ensure model has been trained (check if model coefficients are set)
    assert hasattr(model.model, 'coef_'), "Model coefficients should be set after training."
    assert hasattr(model.model, 'intercept_'), "Model intercept should be set after training."
