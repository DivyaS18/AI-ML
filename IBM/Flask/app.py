import numpy as np
import pickle
import pandas as pd
import os
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the trained model and encoder
model = pickle.load(open('model.pkl', 'rb'))
scale = pickle.load(open('scale.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('My page.html')

@app.route('/predict', methods=["POST", "GET"])
def predict():
    try:
        # Get input features from the form
        input_feature = [float(x) for x in request.form.values()]
        features_values = np.array(input_feature).reshape(1, -1)  # Convert the list to a 2D array for prediction

        # Define feature names
        names = ['holiday', 'temp', 'rain', 'snow', 'weather', 'year', 'month', 'day', 'hours', 'minutes', 'seconds']
        
        # Create a DataFrame from input features
        data = pd.DataFrame(features_values, columns=names)

        # Apply scaling if necessary
        data_scaled = scale.fit_transform(data)

        # Make prediction
        prediction = model.predict(data_scaled)
        print(prediction)

        # Return prediction to HTML page
        prediction_text = f"Estimated Traffic Volume is: {prediction[0]}"
        return render_template("My page.html", prediction_text=prediction_text)
    
    except Exception as e:
        return render_template("My page.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port, debug=True, use_reloader=False)
