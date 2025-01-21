from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model
model = joblib.load('housing_model.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the request
        data = request.get_json()
        X_test = np.array(data['features']).reshape(1, -1)

        # Predict the house price using the model
        prediction = model.predict(X_test)

        # Return the prediction as JSON
        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
