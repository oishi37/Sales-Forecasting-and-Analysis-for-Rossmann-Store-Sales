from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

    
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get data posted as JSON
    # Convert the dictionary to a list of feature values
    features = [data['features'][key] for key in sorted(data['features'].keys())]
    features = np.array(features).reshape(1, -1)  # Convert data to numpy array
    prediction = model.predict(features)  # Make prediction
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
