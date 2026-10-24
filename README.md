# Breast Cancer Prediction  🎗️

A full-stack machine learning web application that predicts whether a breast tumor is benign or malignant based on specific cell nucleus measurements. Built with a custom TensorFlow neural network and served via a responsive Python Flask backend.

## 🚀 Features
* **Machine Learning Model:** Utilizes a Deep Learning Sequential model trained with TensorFlow/Keras.
* **Risk Ratio Analysis:** Outputs both a binary classification (Benign/Malignant) and a precise decimal probability percentage.
* **Strict Input Validation:** Form logic prevents empty submissions and ensures the model only processes valid, user-provided numerical data.
* **Responsive UI:** A clean, dark-mode user interface styled with custom CSS and Bootstrap, designed to stretch dynamically and work perfectly on all screen sizes.
* **Robust Input Handling:** Includes data normalization layers to ensure extreme inputs do not break the prediction mathematical logic.

## 💻 Tech Stack
* **Backend:** Python, Flask
* **Machine Learning:** TensorFlow, Keras, NumPy
* **Frontend:** HTML5, CSS3, JavaScript, Bootstrap

## 📊 Dataset
* **File:** `breast_cancer_data.csv`
* **Description:** Contains the numerical measurements of cell nuclei (Radius, Texture, Perimeter, Area, Smoothness) used as a structural reference and for testing real-world data distributions.

## 📂 Project Structure
* `app.py`: The main Flask server application that handles web requests and interacts with the AI model.
* `train_model.py`: The script used to generate logical training data, build the neural network, and save the `.keras` model.
* `breast_cancer_model.keras`: The compiled and trained TensorFlow model.
* `breast_cancer_data.csv`: The dataset containing tumor measurement parameters.
* `static/index.html`: The frontend user interface containing the asynchronous fetch logic.
## 🏗️ System Architecture Diagram

![System Architecture Workflow](image_7eb398.png)
