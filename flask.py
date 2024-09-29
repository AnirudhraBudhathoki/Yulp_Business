from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the pre-trained model
model = pickle.load(open('model.pkl', 'rb'))

# Initialize the Flask application
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    int_features = [float(x) for x in request.form.values()]
    
    # Handle the input features and placeholders
    placeholder_features = [0] * (59 - len(int_features))
    final_features = np.array([int_features + placeholder_features])
    
    # Make prediction
    prediction = model.predict(final_features)
    
    # Output result
    output = prediction[0]
    if output == 1:
        result = 'The customer is likely to churn.'
    else:
        result = 'The customer is not likely to churn.'

    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)