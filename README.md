# Crop Recommendation System - Flask Backend

A machine learning-powered crop recommendation system that suggests the best crop based on soil and weather conditions. The system uses Random Forest, SVM, or XGBoost models to make predictions.

## Features

- 🌱 AI-powered crop recommendations
- 🌐 Web interface with beautiful UI
- 🔌 REST API for programmatic access
- 📊 Model performance tracking
- 🛡️ Error handling and validation

## Quick Start

### Option 1: One-Command Setup (Recommended)

```bash
python setup_and_run.py
```

This single command will:
- Check for dataset and create a sample if needed
- Install all dependencies
- Train and save the model
- Test the model
- Start the Flask application

### Option 2: Manual Setup

#### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 2. Prepare Dataset

If you have the original dataset, place it as `Crop_recommendation.csv` in the project directory.

If you don't have the dataset, create a sample one:

```bash
python download_sample_data.py
```

#### 3. Train and Save the Model

```bash
python train_and_save_model.py
```

This will:
- Load the crop recommendation dataset
- Apply feature engineering
- Train multiple models (Random Forest, SVM, XGBoost)
- Select the best performing model
- Save the model and preprocessing components to the `models/` directory

#### 4. Test the Model (Optional)

```bash
python test_model.py
```

#### 5. Run the Flask Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

## Usage

### Web Interface

1. Open your browser and go to `http://localhost:5000`
2. Fill in the soil and weather parameters:
   - Nitrogen (N) - ppm
   - Phosphorus (P) - ppm
   - Potassium (K) - ppm
   - Temperature (°C)
   - Humidity (%)
   - pH Level
   - Rainfall (mm)
3. Click "Get Crop Recommendation"
4. View the recommended crop with confidence score

### REST API

#### Predict Crop

**Endpoint:** `POST /predict`

**Request Body:**
```json
{
    "N": 50,
    "P": 30,
    "K": 40,
    "temperature": 25.5,
    "humidity": 65.2,
    "ph": 6.5,
    "rainfall": 120.5
}
```

**Response:**
```json
{
    "predicted_crop": "rice",
    "confidence": 85.2,
    "model_info": {
        "model_name": "XGBoost",
        "accuracy": 0.98
    }
}
```

#### Other Endpoints

- `GET /` - Web interface
- `GET /model_info` - Model information
- `GET /health` - Health check
- `POST /predict_form` - Form submission endpoint

## File Structure

```
├── app.py                          # Flask application
├── train_and_save_model.py         # Model training script
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── templates/                      # HTML templates
│   ├── index.html                  # Main form page
│   ├── result.html                 # Results page
│   └── error.html                  # Error page
└── models/                         # Saved models (created after training)
    ├── crop_recommendation_model.pkl
    ├── scaler.pkl
    ├── label_encoder.pkl
    ├── rainfall_encoder.pkl
    ├── ph_encoder.pkl
    ├── feature_names.pkl
    └── model_info.pkl
```

## Model Information

The system automatically selects the best performing model from:
- **Random Forest Classifier** - Ensemble method with 100 trees
- **Support Vector Machine** - RBF kernel
- **XGBoost Classifier** - Gradient boosting

The selected model and its accuracy are saved and displayed in the API responses.

## Input Parameters

| Parameter | Description | Range | Unit |
|-----------|-------------|-------|------|
| N | Nitrogen content | 0-200 | ppm |
| P | Phosphorus content | 0-200 | ppm |
| K | Potassium content | 0-300 | ppm |
| temperature | Average temperature | -10 to 50 | °C |
| humidity | Relative humidity | 0-100 | % |
| ph | Soil pH level | 0-14 | - |
| rainfall | Annual rainfall | 0-500 | mm |

## Deployment

For production deployment, consider:

1. **Environment Variables:**
   ```bash
   export FLASK_ENV=production
   export FLASK_DEBUG=False
   ```

2. **WSGI Server:**
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

3. **Docker (optional):**
   ```dockerfile
   FROM python:3.9-slim
   COPY . /app
   WORKDIR /app
   RUN pip install -r requirements.txt
   CMD ["python", "app.py"]
   ```

## Troubleshooting

### Common Issues

1. **TemplateNotFound Error:**
   - Ensure the `templates/` directory exists
   - Check that all HTML files are present

2. **Model Loading Error:**
   - Run `train_and_save_model.py` first
   - Ensure the `models/` directory contains all required files

3. **Dataset Not Found:**
   - Place your `Crop_recommendation.csv` file in the project root
   - Or update the path in `train_and_save_model.py`

### Health Check

Visit `http://localhost:5000/health` to check if the system is running properly.

## License

This project is open source and available under the MIT License.
