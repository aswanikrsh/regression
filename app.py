from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained model and scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['features']  # Expecting JSON with 'features' key
        data_array = np.array(data).reshape(1, -1)
        scaled_data = scaler.transform(data_array)
        prediction = model.predict(scaled_data)[0]
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
