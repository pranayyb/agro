# Plant Disease and Crop Prediction API

This project is a FastAPI-based web application that predicts plant diseases from leaf images and offers crop prediction based on agricultural parameters. The app uses machine learning models for both crop and disease predictions and provides suggestions for treatment and fertilizers based on identified plant diseases.

## Features
- **Crop Prediction**: Predict the best crop to grow based on agricultural parameters such as nitrogen, phosphorus, potassium levels, temperature, humidity, etc.
- **Plant Disease Detection**: Detect plant diseases from uploaded leaf images and provide disease-specific recommendations.
- **Fertilizer and Treatment Recommendations**: Get personalized fertilizer and treatment recommendations based on the detected plant disease.

## Technologies Used
- **FastAPI**: A modern, fast web framework for building APIs with Python 3.7+.
- **Uvicorn**: ASGI server to serve the FastAPI app.
- **Pickle**: For loading pre-trained machine learning models.
- **InferenceHTTPClient**: For interacting with the external model API.
- **Pillow (PIL)**: For image handling.

## Requirements
- Python 3.7+
- FastAPI
- Uvicorn
- Pillow
- Numpy
- Scikit-learn (for model loading)
- Inference SDK (for external model inference)

## Setup and Installation

### 1. Clone the Repository

Start by cloning the repository to your local machine:

```bash
git clone https://github.com/pranayyb/agro.git
cd agro
```
### 2. Create a Virtual Environment (Optional but Recommended)

It is highly recommended to use a virtual environment to isolate the project dependencies.

#### Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux/MacOS:
```bash
python3 -m venv venv
source venv/bin/activate
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Start the Application
```bash
uvicorn main:app --reload
```
