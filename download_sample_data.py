import pandas as pd
import numpy as np
import os

def create_sample_dataset():
    """Create a sample crop recommendation dataset based on typical values"""
    
    print("Creating sample crop recommendation dataset...")
    
    # Define crop characteristics based on typical agricultural data
    # All 22 crops from the original dataset
    crop_data = {
        'rice': {
            'N_range': (50, 120), 'P_range': (20, 60), 'K_range': (30, 80),
            'temp_range': (20, 35), 'humidity_range': (60, 90), 'ph_range': (5.5, 7.5),
            'rainfall_range': (100, 300)
        },
        'maize': {
            'N_range': (60, 140), 'P_range': (25, 70), 'K_range': (40, 90),
            'temp_range': (18, 30), 'humidity_range': (50, 85), 'ph_range': (5.5, 7.5),
            'rainfall_range': (80, 200)
        },
        'chickpea': {
            'N_range': (20, 60), 'P_range': (10, 40), 'K_range': (20, 50),
            'temp_range': (15, 30), 'humidity_range': (40, 70), 'ph_range': (6.0, 8.0),
            'rainfall_range': (40, 120)
        },
        'kidneybeans': {
            'N_range': (30, 80), 'P_range': (15, 45), 'K_range': (25, 60),
            'temp_range': (18, 28), 'humidity_range': (50, 80), 'ph_range': (6.0, 7.5),
            'rainfall_range': (60, 150)
        },
        'pigeonpeas': {
            'N_range': (25, 70), 'P_range': (12, 40), 'K_range': (20, 55),
            'temp_range': (20, 32), 'humidity_range': (45, 75), 'ph_range': (6.0, 8.0),
            'rainfall_range': (50, 140)
        },
        'mothbeans': {
            'N_range': (20, 50), 'P_range': (8, 30), 'K_range': (15, 40),
            'temp_range': (22, 35), 'humidity_range': (40, 70), 'ph_range': (6.5, 8.5),
            'rainfall_range': (30, 100)
        },
        'mungbean': {
            'N_range': (25, 60), 'P_range': (10, 35), 'K_range': (18, 45),
            'temp_range': (20, 30), 'humidity_range': (45, 75), 'ph_range': (6.0, 7.5),
            'rainfall_range': (40, 120)
        },
        'blackgram': {
            'N_range': (30, 70), 'P_range': (12, 40), 'K_range': (20, 50),
            'temp_range': (22, 32), 'humidity_range': (50, 80), 'ph_range': (6.0, 8.0),
            'rainfall_range': (50, 130)
        },
        'lentil': {
            'N_range': (20, 50), 'P_range': (8, 30), 'K_range': (15, 40),
            'temp_range': (15, 25), 'humidity_range': (40, 70), 'ph_range': (6.0, 8.0),
            'rainfall_range': (40, 100)
        },
        'pomegranate': {
            'N_range': (40, 100), 'P_range': (20, 60), 'K_range': (30, 80),
            'temp_range': (20, 35), 'humidity_range': (50, 80), 'ph_range': (6.0, 8.0),
            'rainfall_range': (60, 200)
        },
        'banana': {
            'N_range': (60, 120), 'P_range': (25, 70), 'K_range': (40, 100),
            'temp_range': (22, 35), 'humidity_range': (60, 90), 'ph_range': (5.5, 7.5),
            'rainfall_range': (100, 300)
        },
        'mango': {
            'N_range': (50, 100), 'P_range': (20, 60), 'K_range': (30, 80),
            'temp_range': (20, 35), 'humidity_range': (50, 85), 'ph_range': (5.5, 8.0),
            'rainfall_range': (80, 250)
        },
        'grapes': {
            'N_range': (40, 90), 'P_range': (15, 50), 'K_range': (25, 70),
            'temp_range': (15, 30), 'humidity_range': (40, 75), 'ph_range': (6.0, 7.5),
            'rainfall_range': (50, 150)
        },
        'watermelon': {
            'N_range': (30, 80), 'P_range': (15, 45), 'K_range': (25, 60),
            'temp_range': (20, 35), 'humidity_range': (50, 80), 'ph_range': (6.0, 7.5),
            'rainfall_range': (60, 180)
        },
        'muskmelon': {
            'N_range': (25, 70), 'P_range': (12, 40), 'K_range': (20, 55),
            'temp_range': (18, 32), 'humidity_range': (45, 75), 'ph_range': (6.0, 7.5),
            'rainfall_range': (50, 150)
        },
        'apple': {
            'N_range': (30, 80), 'P_range': (15, 50), 'K_range': (25, 70),
            'temp_range': (5, 25), 'humidity_range': (40, 80), 'ph_range': (6.0, 7.5),
            'rainfall_range': (60, 200)
        },
        'orange': {
            'N_range': (40, 90), 'P_range': (20, 60), 'K_range': (30, 80),
            'temp_range': (15, 30), 'humidity_range': (50, 85), 'ph_range': (5.5, 7.5),
            'rainfall_range': (80, 250)
        },
        'papaya': {
            'N_range': (50, 100), 'P_range': (20, 60), 'K_range': (30, 80),
            'temp_range': (22, 35), 'humidity_range': (60, 90), 'ph_range': (5.5, 7.5),
            'rainfall_range': (100, 300)
        },
        'coconut': {
            'N_range': (40, 100), 'P_range': (20, 60), 'K_range': (30, 80),
            'temp_range': (22, 35), 'humidity_range': (60, 90), 'ph_range': (5.5, 8.0),
            'rainfall_range': (100, 300)
        },
        'cotton': {
            'N_range': (80, 150), 'P_range': (30, 80), 'K_range': (50, 100),
            'temp_range': (20, 35), 'humidity_range': (40, 70), 'ph_range': (5.5, 8.0),
            'rainfall_range': (60, 180)
        },
        'jute': {
            'N_range': (60, 120), 'P_range': (25, 70), 'K_range': (40, 90),
            'temp_range': (20, 35), 'humidity_range': (60, 90), 'ph_range': (6.0, 8.0),
            'rainfall_range': (100, 300)
        },
        'coffee': {
            'N_range': (40, 100), 'P_range': (20, 60), 'K_range': (30, 80),
            'temp_range': (15, 28), 'humidity_range': (60, 90), 'ph_range': (5.5, 7.0),
            'rainfall_range': (100, 300)
        }
    }
    
    # Generate data for each crop
    all_data = []
    samples_per_crop = 200
    
    for crop, ranges in crop_data.items():
        for _ in range(samples_per_crop):
            # Add some randomness to make it more realistic
            noise_factor = 0.1
            
            N = np.random.uniform(ranges['N_range'][0], ranges['N_range'][1])
            P = np.random.uniform(ranges['P_range'][0], ranges['P_range'][1])
            K = np.random.uniform(ranges['K_range'][0], ranges['K_range'][1])
            temperature = np.random.uniform(ranges['temp_range'][0], ranges['temp_range'][1])
            humidity = np.random.uniform(ranges['humidity_range'][0], ranges['humidity_range'][1])
            ph = np.random.uniform(ranges['ph_range'][0], ranges['ph_range'][1])
            rainfall = np.random.uniform(ranges['rainfall_range'][0], ranges['rainfall_range'][1])
            
            # Add some noise
            N += np.random.normal(0, N * noise_factor)
            P += np.random.normal(0, P * noise_factor)
            K += np.random.normal(0, K * noise_factor)
            temperature += np.random.normal(0, temperature * noise_factor)
            humidity += np.random.normal(0, humidity * noise_factor)
            ph += np.random.normal(0, ph * noise_factor)
            rainfall += np.random.normal(0, rainfall * noise_factor)
            
            # Ensure values are within reasonable bounds
            N = max(0, min(200, N))
            P = max(0, min(200, P))
            K = max(0, min(300, K))
            temperature = max(0, min(50, temperature))
            humidity = max(0, min(100, humidity))
            ph = max(0, min(14, ph))
            rainfall = max(0, min(500, rainfall))
            
            all_data.append({
                'N': round(N, 1),
                'P': round(P, 1),
                'K': round(K, 1),
                'temperature': round(temperature, 1),
                'humidity': round(humidity, 1),
                'ph': round(ph, 1),
                'rainfall': round(rainfall, 1),
                'label': crop
            })
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save to CSV
    df.to_csv('Crop_recommendation.csv', index=False)
    
    print(f"Sample dataset created with {len(df)} samples")
    print(f"Dataset shape: {df.shape}")
    print(f"Crops included: {df['label'].unique()}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    print("\nDataset saved as 'Crop_recommendation.csv'")
    
    return df

if __name__ == "__main__":
    create_sample_dataset()
