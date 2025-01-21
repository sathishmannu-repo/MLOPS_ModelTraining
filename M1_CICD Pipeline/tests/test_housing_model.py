import pytest
import numpy as np
from housing_model import HousingModel

def test_housing_model_train_and_predict():
    # Create mock training data
    X_train = np.array([[1], [2], [3], [4], [5]])
    y_train = np.array([1, 2, 3, 4, 5])

    # Create mock test data
    X_test = np.array([[6], [7]])

    # Expected predictions
    expected_predictions = np.array([6, 7])

    # Initialize and train the model
    model = HousingModel()
    model.train(X_train, y_train)

    # Predict and assert
    predictions = model.predict(X_test)
    np.testing.assert_almost_equal(predictions, expected_predictions, decimal=5)
