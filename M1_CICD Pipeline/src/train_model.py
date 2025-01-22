import pandas as pd
import math
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from housing_model import HousingModel  # Make sure the path is correct


# File path
file_path = 'M1_CICD Pipeline/data/housing.csv'

# Load the dataset
data = pd.read_csv(file_path)

# Prepare features and target variable
X = data.drop(columns=['PRICE'])
y = data['PRICE']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and train the model
model = HousingModel()
model.train(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = math.sqrt(mse)
r2 = r2_score(y_test, predictions)

# Print evaluation results
print("Model trained successfully.")  # No need for f-string here
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² (Coefficient of Determination): {r2:.2f}")

# Save the trained model to a file
model_file = 'M1_CICD Pipeline/model/housing_model.pkl'
joblib.dump(model.model, model_file)
print(f"Model saved to {model_file}")
