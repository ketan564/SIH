import pandas as pd
import numpy as np
import joblib
import os

def test_model():
    """Test the saved model with sample data"""
    
    # Check if models exist
    if not os.path.exists('models/crop_recommendation_model.pkl'):
        print("Models not found! Please run train_and_save_model.py first.")
        return
    
    # Load the model and preprocessing components
    print("Loading model and preprocessing components...")
    model = joblib.load('models/crop_recommendation_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    label_encoder = joblib.load('models/label_encoder.pkl')
    rainfall_encoder = joblib.load('models/rainfall_encoder.pkl')
    ph_encoder = joblib.load('models/ph_encoder.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    model_info = joblib.load('models/model_info.pkl')
    
    print(f"Model: {model_info['model_name']}")
    print(f"Accuracy: {model_info['accuracy']:.4f}")
    print(f"Feature names: {feature_names}")
    print(f"Label classes: {model_info['label_classes']}")
    
    # Test with sample data
    test_samples = [
        {
            'N': 90, 'P': 42, 'K': 43, 'temperature': 20.9, 
            'humidity': 82.0, 'ph': 6.5, 'rainfall': 202.9
        },
        {
            'N': 50, 'P': 30, 'K': 40, 'temperature': 25.5, 
            'humidity': 65.2, 'ph': 6.5, 'rainfall': 120.5
        },
        {
            'N': 120, 'P': 60, 'K': 80, 'temperature': 30.0, 
            'humidity': 70.0, 'ph': 7.0, 'rainfall': 150.0
        }
    ]
    
    def preprocess_input(data):
        """Preprocess input data for prediction"""
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
    
    print("\nTesting model with sample data:")
    print("=" * 50)
    
    for i, sample in enumerate(test_samples, 1):
        print(f"\nTest Sample {i}:")
        print(f"Input: {sample}")
        
        try:
            # Preprocess the input
            processed_data = preprocess_input(sample)
            
            # Make prediction
            prediction = model.predict(processed_data)[0]
            predicted_crop = label_encoder.inverse_transform([prediction])[0]
            
            # Get prediction probability (if available)
            try:
                prediction_proba = model.predict_proba(processed_data)[0]
                confidence = max(prediction_proba) * 100
            except:
                confidence = None
            
            print(f"Predicted Crop: {predicted_crop}")
            if confidence:
                print(f"Confidence: {confidence:.1f}%")
            
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_model()
