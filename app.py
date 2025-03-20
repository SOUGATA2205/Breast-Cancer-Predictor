from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__, template_folder='templates')


# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
    
print("Model loaded successfully!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get feature values from the form
    features = [
        float(request.form.get('radius_mean')),
        float(request.form.get('texture_mean')),
        float(request.form.get('perimeter_mean')),
        float(request.form.get('area_mean')),
        float(request.form.get('smoothness_mean')),
        float(request.form.get('compactness_mean')),
        float(request.form.get('concavity_mean')),
        float(request.form.get('concave_points_mean')),
        float(request.form.get('symmetry_mean')),
        float(request.form.get('fractal_dimension_mean')),
        float(request.form.get('radius_se')),
        float(request.form.get('texture_se')),
        float(request.form.get('perimeter_se')),
        float(request.form.get('area_se')),
        float(request.form.get('smoothness_se')),
        float(request.form.get('compactness_se')),
        float(request.form.get('concavity_se')),
        float(request.form.get('concave_points_se')),
        float(request.form.get('symmetry_se')),
        float(request.form.get('fractal_dimension_se')),
        float(request.form.get('radius_worst')),
        float(request.form.get('texture_worst')),
        float(request.form.get('perimeter_worst')),
        float(request.form.get('area_worst')),
        float(request.form.get('smoothness_worst')),
        float(request.form.get('compactness_worst')),
        float(request.form.get('concavity_worst')),
        float(request.form.get('concave_points_worst')),
        float(request.form.get('symmetry_worst')),
        float(request.form.get('fractal_dimension_worst'))
    ]
    
    # Add 0 for the ID feature since it's in the model but not used for prediction
    final_features = [0] + features  # Add a placeholder for the 'id' column
    
    # Make prediction
    prediction = model.predict([final_features])
    probability = model.predict_proba([final_features])
    
    # Determine result
    result = "Malignant (M)" if prediction[0] == 'M' else "Benign (B)"
    confidence = probability[0][1] if prediction[0] == 'M' else probability[0][0]
    
    return render_template('index.html', 
                          prediction_text=f'Breast Cancer Prediction: {result}',
                          confidence=f'Confidence: {confidence:.2%}')

if __name__ == '__main__':
    app.run(debug=True)