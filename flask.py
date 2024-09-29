from flask import Flask, render_template, request
import pickle
import numpy as np

# Load models
DT_model = pickle.load(open('DT_model.pkl', 'rb'))
RF_model = pickle.load(open('RF_model.pkl', 'rb'))
Ada_model = pickle.load(open('Ada_model.pkl', 'rb'))
xgb_model = pickle.load(open('xgb_model1.pkl', 'rb'))

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # HTML page to display the form

@app.route('/predict', methods=['POST'])
def predict():
    model_option = request.form['model_option']  # Get model option from form

    # Get features from the form
    Feature1 = float(request.form['tip_count'])
    Feature2 = float(request.form['review_count_y'])
    Feature3 = float(request.form['avg_review_rating'])
    Feature4 = float(request.form['longitude'])
    Feature5 = float(request.form['latitude'])
    Feature6 = float(request.form['review_count_x'])
    Feature7 = float(request.form['checkin_count'])
    Feature8 = float(request.form['stars'])

    # Prepare features array
    features = np.array([[Feature1, Feature2, Feature3, Feature4, Feature5, Feature6, Feature7, Feature8]])

    # Select the model based on user input
    if model_option == 'Decision Tree':
        model = DT_model
    elif model_option == 'Random Forest':
        model = RF_model
    elif model_option == 'AdaBoost':
        model = Ada_model
    elif model_option == 'XGBoost':
        model = xgb_model
    else:
        return "Invalid model option selected."

    # Prediction
    prediction = model.predict(features)
    if prediction[0] == 1:
        result = 'The Business is likely to open.'
    else:
        result = 'The Business is not likely to open.'

    return render_template('result.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)