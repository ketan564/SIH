from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Global variables to store loaded models
model = None
scaler = None
label_encoder = None
rainfall_encoder = None
ph_encoder = None
feature_names = None
model_info = None

def load_models():
    """Load all saved models and preprocessing components"""
    global model, scaler, label_encoder, rainfall_encoder, ph_encoder, feature_names, model_info
    
    try:
        # Load the main model
        model = joblib.load('models/crop_recommendation_model.pkl')
        
        # Load preprocessing components
        scaler = joblib.load('models/scaler.pkl')
        label_encoder = joblib.load('models/label_encoder.pkl')
        rainfall_encoder = joblib.load('models/rainfall_encoder.pkl')
        ph_encoder = joblib.load('models/ph_encoder.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        model_info = joblib.load('models/model_info.pkl')
        
        print("All models loaded successfully!")
        print(f"Model: {model_info['model_name']}")
        print(f"Accuracy: {model_info['accuracy']:.4f}")
        
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        raise e

def preprocess_input(data):
    """Preprocess input data for prediction"""
    try:
        # Create a DataFrame with the input data
        input_df = pd.DataFrame([data])
        
        # Apply feature engineering (same as training)
        input_df['NPK'] = (input_df['N'] + input_df['P'] + input_df['K']) / 3
        input_df['THI'] = input_df['temperature'] * input_df['humidity'] / 100
        
        # Categorize rainfall using pd.cut (same as training)
        rainfall_levels = pd.cut(input_df['rainfall'],
                                bins=[0, 50, 100, 200, 300],
                                labels=['Low', 'Medium', 'High', 'Very High'])
        input_df['rainfall_level'] = rainfall_levels
        
        # Categorize pH (same as training)
        def ph_category(p):
            if p < 5.5:
                return 'Acidic'
            elif p <= 7.5:
                return 'Neutral'
            else:
                return 'Alkaline'
        
        input_df['ph_category'] = input_df['ph'].apply(ph_category)
        input_df['temp_rain_interaction'] = input_df['temperature'] * input_df['rainfall']
        input_df['ph_rain_interaction'] = input_df['ph'] * input_df['rainfall']
        
        # Encode categorical variables
        input_df['rainfall_level'] = rainfall_encoder.transform(input_df['rainfall_level'])
        input_df['ph_category'] = ph_encoder.transform(input_df['ph_category'])
        
        # Ensure all required features are present and in correct order
        for feature in feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        # Reorder columns to match training data
        input_df = input_df[feature_names]
        
        # Scale the features
        scaled_data = scaler.transform(input_df)
        
        return scaled_data
        
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        raise e

@app.route('/')
def home():
    """Home page with a simple form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for crop prediction"""
    try:
        # Get input data from request
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Preprocess the input data
        processed_data = preprocess_input(data)
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        
        # Get prediction probability (if available)
        try:
            prediction_proba = model.predict_proba(processed_data)[0]
            confidence = max(prediction_proba) * 100
        except:
            confidence = None
        
        # Decode the prediction
        predicted_crop = label_encoder.inverse_transform([prediction])[0]
        
        # Prepare response
        response = {
            'predicted_crop': predicted_crop,
            'confidence': confidence,
            'model_info': {
                'model_name': model_info['model_name'],
                'accuracy': model_info['accuracy']
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/predict_form', methods=['POST'])
def predict_form():
    """Handle form submission for web interface"""
    try:
        # Get form data
        data = {
            'N': float(request.form['N']),
            'P': float(request.form['P']),
            'K': float(request.form['K']),
            'temperature': float(request.form['temperature']),
            'humidity': float(request.form['humidity']),
            'ph': float(request.form['ph']),
            'rainfall': float(request.form['rainfall'])
        }
        
        # Preprocess the input data
        processed_data = preprocess_input(data)
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        
        # Get prediction probability (if available)
        try:
            prediction_proba = model.predict_proba(processed_data)[0]
            confidence = max(prediction_proba) * 100
        except:
            confidence = None
        
        # Decode the prediction
        predicted_crop = label_encoder.inverse_transform([prediction])[0]
        
        return render_template('result.html', 
                             predicted_crop=predicted_crop, 
                             confidence=confidence,
                             input_data=data)
        
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/model_info')
def get_model_info():
    """Get information about the loaded model"""
    try:
        return jsonify(model_info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    # Load models when starting the app
    load_models()
    
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    print("Starting Flask app...")
    print("API endpoints:")
    print("- GET  / : Home page with form")
    print("- POST /predict : JSON API for predictions")
    print("- POST /predict_form : Form submission endpoint")
    print("- GET  /model_info : Model information")
    print("- GET  /health : Health check")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

