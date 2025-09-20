import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import joblib
import os

def feature_engineer(data):
    """Feature engineering function from the notebook"""
    data['NPK'] = (data['N'] + data['P'] + data['K']) / 3
    data['THI'] = data['temperature'] * data['humidity'] / 100
    data['rainfall_level'] = pd.cut(data['rainfall'],
                              bins=[0, 50, 100, 200, 300],
                              labels=['Low', 'Medium', 'High', 'Very High'])
    
    def ph_category(p):
        if p < 5.5:
            return 'Acidic'
        elif p <= 7.5:
            return 'Neutral'
        else:
            return 'Alkaline'
    
    data['ph_category'] = data['ph'].apply(ph_category)
    data['temp_rain_interaction'] = data['temperature'] * data['rainfall']
    data['ph_rain_interaction'] = data['ph'] * data['rainfall']

    return data

def train_and_save_models():
    """Train models and save the best one for Flask deployment"""
    
    # Load dataset
    print("Loading dataset...")
    
    try:
        data = pd.read_csv('Crop_recommendation.csv')
        print("✅ Real dataset loaded successfully!")
        print(f"Dataset shape: {data.shape}")
        print(f"Unique crops: {len(data['label'].unique())}")
        print(f"Crops: {data['label'].unique()}")
    except FileNotFoundError:
        print("❌ Dataset not found. Please ensure 'Crop_recommendation.csv' is in the current directory.")
        print("Creating a sample dataset for demonstration...")
        # Create a small sample dataset for demonstration
        np.random.seed(42)
        n_samples = 1000
        data = pd.DataFrame({
            'N': np.random.randint(0, 150, n_samples),
            'P': np.random.randint(5, 150, n_samples),
            'K': np.random.randint(5, 210, n_samples),
            'temperature': np.random.uniform(8.8, 43.7, n_samples),
            'humidity': np.random.uniform(14.3, 99.9, n_samples),
            'ph': np.random.uniform(3.5, 9.9, n_samples),
            'rainfall': np.random.uniform(20.2, 298.6, n_samples),
            'label': np.random.choice(['rice', 'wheat', 'maize', 'cotton', 'sugarcane'], n_samples)
        })
    
    print(f"Dataset shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")
    print(f"Label distribution:\n{data['label'].value_counts()}")
    
    # Feature engineering - apply to the data
    print("Applying feature engineering...")
    data = feature_engineer(data)
    
    # Encoding categorical columns - following the original notebook pattern
    print("Encoding categorical variables...")
    le_label = LabelEncoder()
    le_rainfall = LabelEncoder()
    le_ph = LabelEncoder()
    
    # Apply encoding to the feature-engineered data
    data['label'] = le_label.fit_transform(data['label'])
    data['rainfall_level'] = le_rainfall.fit_transform(data['rainfall_level'])
    data['ph_category'] = le_ph.fit_transform(data['ph_category'])
    
    # Prepare features and target
    X = data.drop('label', axis=1)
    y = data['label']
    
    # Split the data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature scaling
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    print("Training models...")
    
    # Random Forest
    print("Training Random Forest...")
    rfc = RandomForestClassifier(n_estimators=100, random_state=42)
    rfc.fit(X_train_scaled, y_train)
    rfc_pred = rfc.predict(X_test_scaled)
    rfc_accuracy = accuracy_score(y_test, rfc_pred)
    print(f"Random Forest Accuracy: {rfc_accuracy:.4f}")
    
    # SVM
    print("Training SVM...")
    svm = SVC(kernel='rbf', random_state=42)
    svm.fit(X_train_scaled, y_train)
    svm_pred = svm.predict(X_test_scaled)
    svm_accuracy = accuracy_score(y_test, svm_pred)
    print(f"SVM Accuracy: {svm_accuracy:.4f}")
    
    # XGBoost
    print("Training XGBoost...")
    xgb = XGBClassifier(random_state=42)
    xgb.fit(X_train_scaled, y_train)
    xgb_pred = xgb.predict(X_test_scaled)
    xgb_accuracy = accuracy_score(y_test, xgb_pred)
    print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")
    
    # Select best model
    models = {
        'RandomForest': (rfc, rfc_accuracy),
        'SVM': (svm, svm_accuracy),
        'XGBoost': (xgb, xgb_accuracy)
    }
    
    best_model_name = max(models.keys(), key=lambda k: models[k][1])
    best_model = models[best_model_name][0]
    
    print(f"\nBest model: {best_model_name} with accuracy: {models[best_model_name][1]:.4f}")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save the best model
    print(f"Saving {best_model_name} model...")
    joblib.dump(best_model, 'models/crop_recommendation_model.pkl')
    
    # Save preprocessing components
    print("Saving preprocessing components...")
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(le_label, 'models/label_encoder.pkl')
    joblib.dump(le_rainfall, 'models/rainfall_encoder.pkl')
    joblib.dump(le_ph, 'models/ph_encoder.pkl')
    
    # Save feature names for reference
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, 'models/feature_names.pkl')
    
    # Save model metadata
    model_info = {
        'model_name': best_model_name,
        'accuracy': models[best_model_name][1],
        'feature_names': feature_names,
        'label_classes': le_label.classes_.tolist(),
        'rainfall_levels': le_rainfall.classes_.tolist(),
        'ph_categories': le_ph.classes_.tolist()
    }
    joblib.dump(model_info, 'models/model_info.pkl')
    
    print("\nModel and preprocessing components saved successfully!")
    print("Files saved in 'models' directory:")
    print("- crop_recommendation_model.pkl")
    print("- scaler.pkl")
    print("- label_encoder.pkl")
    print("- rainfall_encoder.pkl")
    print("- ph_encoder.pkl")
    print("- feature_names.pkl")
    print("- model_info.pkl")
    
    return best_model_name, models[best_model_name][1]

if __name__ == "__main__":
    train_and_save_models()
