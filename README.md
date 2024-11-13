# Plant Disease Detection API

This project is a FastAPI-based web application that detects plant diseases from uploaded images and provides fertilizer and treatment recommendations based on the disease identified. The API leverages a machine learning model (`plants-final/1`) to classify plant diseases from images and offers suggestions to help manage and treat the plants effectively.

## Features

- **Disease Prediction**: Detect plant diseases from images of plant leaves.
- **Fertilizer Recommendations**: Get personalized fertilizer recommendations based on the identified disease.
- **Treatment Recommendations**: Receive suggested treatments, including fungicides and other management strategies.
- **Upload Plant Images**: Users can upload plant images to the API for analysis.

## Technologies Used

- **FastAPI**: A modern, fast web framework for building APIs with Python 3.7+.
- **Uvicorn**: ASGI server used to serve the FastAPI app.
- **Machine Learning Model**: A pre-trained model for detecting plant diseases.

## Requirements

- Python 3.7+
- FastAPI
- Uvicorn
- Pillow (for image handling)
- shutil (for file operations)

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
