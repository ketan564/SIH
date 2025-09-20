import requests
import json

def test_api():
    """Test the Flask API"""
    
    # Test health endpoint
    print("Testing health endpoint...")
    try:
        response = requests.get('http://localhost:5000/health')
        print(f"Health check - Status: {response.status_code}")
        print(f"Health check - Response: {response.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")
    
    print("\n" + "="*50)
    
    # Test prediction endpoint
    print("Testing prediction endpoint...")
    test_data = {
        'N': 90,
        'P': 42,
        'K': 43,
        'temperature': 20.9,
        'humidity': 82.0,
        'ph': 6.5,
        'rainfall': 202.9
    }
    
    try:
        response = requests.post('http://localhost:5000/predict', json=test_data)
        print(f"Prediction - Status: {response.status_code}")
        print(f"Prediction - Response: {response.json()}")
    except Exception as e:
        print(f"Prediction failed: {e}")
    
    print("\n" + "="*50)
    
    # Test model info endpoint
    print("Testing model info endpoint...")
    try:
        response = requests.get('http://localhost:5000/model_info')
        print(f"Model info - Status: {response.status_code}")
        model_info = response.json()
        print(f"Model: {model_info['model_name']}")
        print(f"Accuracy: {model_info['accuracy']:.4f}")
        print(f"Number of crops: {len(model_info['label_classes'])}")
    except Exception as e:
        print(f"Model info failed: {e}")

if __name__ == "__main__":
    test_api()

