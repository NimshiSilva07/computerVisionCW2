# Emotion Recognition Web App

A simple web application for detecting emotions from facial images using **FastAPI** and **TensorFlow/Keras**. Upload a face image to predict one of seven emotions: *angry*, *disgust*, *fear*, *happy*, *neutral*, *sad*, or *surprise*.

## ✨ Features

- Detects **seven emotions**: angry, disgust, fear, happy, neutral, sad, surprise.
- **Responsive UI**: Built with Bootstrap 5 for a clean, mobile-friendly interface.
- **Real-time Preview**: View uploaded images before prediction.
- **CNN-Powered**: Uses a pre-trained convolutional neural network (`emotion_cnn_model.h5`).

## 📂 Project Structure
project/
│
├── app.py                     # FastAPI backend
├── emotion_cnn_model.h5       # Trained CNN model
├── templates/
│   └── index.html             # Frontend HTML template
├── static/                    # CSS, JS, and images (optional)
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
text## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd project

Create a virtual environment:
bashpython -m venv venv

Activate the virtual environment:

Windows:
bashvenv\Scripts\activate

Linux/Mac:
bashsource venv/bin/activate



Install dependencies:
bashpip install -r requirements.txt


Example requirements.txt
textfastapi
uvicorn
tensorflow
numpy
pillow
jinja2
🚀 Running the App

Start the FastAPI server:
bashuvicorn app:app --reload

Open your browser and navigate to:
texthttp://127.0.0.1:8000

Upload a face image and click Predict Emotion to see the result.

🎯 Usage

Visit the web page in your browser.
Upload an image with a single face.
Wait briefly for the prediction.
View the predicted emotion and confidence score below the upload button.


Note: For best results, use images with a single, clearly visible face.

## 🧠 Model Details

Model: Custom CNN trained on the Facial Expression Recognition Dataset.
Input: 48x48 RGB images.
Output: 7 emotion classes (angry, disgust, fear, happy, neutral, sad, surprise).
Preprocessing: Normalization, resizing, and data augmentation.

## ⚠️ Challenges

Hardware Constraints: Limited dataset and batch sizes due to low-spec machines.
Model Performance: MobileNetV2 was too heavy for real-time local predictions.
Image Processing: Ensuring proper resizing and preprocessing of uploaded images.

## 🔮 Future Improvements

Support for detecting emotions from multiple faces.
Deploy with GPU support for faster inference.
Add live webcam-based emotion detection.
Include real-time charts for emotion probabilities.
