from flask import jsonify, Flask, request
import joblib as jb

app = Flask(__name__)

model = jb.load("creditCardFraudDetectionModel.pkl")  # loading the model

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Get the incoming JSON data        
        if not data or 'features' not in data:  # Validate the structure of the request
            return jsonify({'error': 'Invalid input, "features" are required'}), 400
        
        features = data['features']  # Extract the features
        predictionResults = model.predict([features])  # Use the features to make a prediction
        return jsonify({'fraud': bool(predictionResults[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 500  # Handle any unexpected errors

if __name__ == "__main__":
    app.run(debug=True)
