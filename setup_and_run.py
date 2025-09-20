#!/usr/bin/env python3
"""
Complete setup script for the Crop Recommendation Flask App
This script will:
1. Check for dataset and create sample if needed
2. Train and save the model
3. Test the model
4. Start the Flask app
"""

import os
import sys
import subprocess
import pandas as pd

def check_dataset():
    """Check if dataset exists"""
    if os.path.exists('Crop_recommendation.csv'):
        print("✅ Dataset found: Crop_recommendation.csv")
        # Check if it has all 22 crops
        try:
            import pandas as pd
            data = pd.read_csv('Crop_recommendation.csv')
            unique_crops = len(data['label'].unique())
            print(f"✅ Dataset contains {unique_crops} unique crops")
            if unique_crops == 22:
                print("✅ Perfect! All 22 crops are present")
            else:
                print(f"⚠️  Warning: Expected 22 crops, found {unique_crops}")
        except Exception as e:
            print(f"⚠️  Warning: Could not verify dataset contents: {e}")
        return True
    else:
        print("❌ Dataset not found. Please ensure 'Crop_recommendation.csv' is in the current directory.")
        print("You can create a sample dataset by running: python download_sample_data.py")
        return False

# def install_requirements():
#     """Install required packages"""
#     print("📦 Installing requirements...")
#     try:
#         subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
#         print("✅ Requirements installed successfully!")
#         return True
#     except Exception as e:
#         print(f"❌ Error installing requirements: {e}")
#         return False

def train_model():
    """Train and save the model"""
    print("🤖 Training model...")
    try:
        from train_and_save_model import train_and_save_models
        model_name, accuracy = train_and_save_models()
        print(f"✅ Model trained successfully! ({model_name}, Accuracy: {accuracy:.4f})")
        return True
    except Exception as e:
        print(f"❌ Error training model: {e}")
        return False

def test_model():
    """Test the trained model"""
    print("🧪 Testing model...")
    try:
        from test_model import test_model
        test_model()
        print("✅ Model test completed!")
        return True
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False

def start_flask_app():
    """Start the Flask application"""
    print("🚀 Starting Flask application...")
    print("=" * 50)
    print("🌱 Crop Recommendation System")
    print("=" * 50)
    print("📱 Web Interface: http://localhost:5000")
    print("🔌 API Endpoint: http://localhost:5000/predict")
    print("📊 Model Info: http://localhost:5000/model_info")
    print("❤️  Health Check: http://localhost:5000/health")
    print("=" * 50)
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting Flask app: {e}")

def main():
    """Main setup and run function"""
    print("🌱 Crop Recommendation System Setup")
    print("=" * 40)
    
    # Step 1: Check dataset
    if not check_dataset():
        print("❌ Setup failed: Could not create dataset")
        return
    
    # Step 2: Install requirements
    # if not install_requirements():
    #     print("❌ Setup failed: Could not install requirements")
    #     return
    
    # Step 3: Train model
    if not train_model():
        print("❌ Setup failed: Could not train model")
        return
    
    # Step 4: Test model
    if not test_model():
        print("❌ Setup failed: Model test failed")
        return
    
    print("\n✅ Setup completed successfully!")
    print("🚀 Starting Flask application...")
    
    # Step 5: Start Flask app
    start_flask_app()

if __name__ == "__main__":
    main()
