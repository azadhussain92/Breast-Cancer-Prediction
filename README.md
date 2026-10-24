# 🎗️ Breast Cancer Prediction

A full-stack machine learning web application that predicts whether a breast tumor is **Benign or Malignant** based on cell nucleus measurements. The application uses a TensorFlow/Keras neural network and a Python Flask backend to process user inputs and return predictions through a responsive web interface.

## 🚀 Features

- **Machine Learning Model:** Uses a deep learning Sequential model built with TensorFlow/Keras.
- **Tumor Classification:** Predicts whether a tumor is benign or malignant.
- **Prediction Probability:** Displays the model's predicted probability alongside the classification.
- **Input Validation:** Validates numerical inputs and prevents empty form submissions.
- **Responsive UI:** Provides a dark-mode interface built with HTML, CSS, JavaScript, and Bootstrap.
- **Data Preprocessing:** Supports numerical feature preprocessing and normalization consistent with the model's training pipeline.
- **Flask Backend:** Connects the web interface with the trained machine learning model.

## 💻 Tech Stack

| Technology | Purpose |
|---|---|
| Python | Backend programming |
| Flask | Web application framework |
| TensorFlow | Deep learning framework |
| Keras | Neural network development |
| NumPy | Numerical computations |
| HTML5 | Web page structure |
| CSS3 | User interface styling |
| JavaScript | Client-side interaction and asynchronous requests |
| Bootstrap | Responsive layout and UI components |

## 📊 Dataset

**Dataset file:** `breast_cancer_data.csv`

The dataset contains numerical measurements of cell nuclei, which may include:

- Radius
- Texture
- Perimeter
- Area
- Smoothness

These measurements are used as model input features. The exact feature names and order must match the dataset and model training pipeline.

### System Architecture Components

**1. Dataset and Data Preprocessing**

- Load the tumor measurement dataset.
- Select the required numerical features.
- Prepare the input data.
- Apply the same preprocessing and normalization used during model training.

**2. Model Training**

- Build a Sequential neural network using TensorFlow/Keras.
- Train the model using the prepared dataset.
- Evaluate the trained model.
- Save the model as `breast_cancer_model.keras`.

**3. Flask Backend**

- Start the Flask web server.
- Receive measurement values from the frontend.
- Validate and preprocess the submitted data.
- Load or access the trained model.
- Generate a prediction.

**4. Prediction and Result Display**

- Process the model's output.
- Determine the predicted class using the configured classification threshold.
- Display the predicted class and probability on the web interface.

> **Note:** The probability produced by a neural network is a model output and should not automatically be interpreted as a clinically validated cancer risk.

## 🔄 Project Workflow

```text
       Breast Cancer Dataset
                 |
                 v
       Data Preprocessing
                 |
                 v
       Train Neural Network
       (TensorFlow / Keras)
                 |
                 v
       Evaluate the Model
                 |
                 v
       Save Trained Model
    breast_cancer_model.keras
                 |
                 v
          Flask Backend
                 |
                 v
       User Enters Measurements
                 |
                 v
        Validate User Inputs
                 |
                 v
      Preprocess Input Features
                 |
                 v
       Trained Model Prediction
                 |
                 v
       Benign / Malignant Result
                 |
                 v
      Display Prediction to User
```

## 📂 Project Structure

```text
Breast-Cancer-Prediction/
│
├── app.py
├── train_model.py
├── breast_cancer_data.csv
├── breast_cancer_model.keras
│
├── static/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── requirements.txt
├── README.md
└── docs/
    └── system_architecture.png
```

*This is a suggested structure. Keep the filenames and folders consistent with your actual project.*
## 📈 Model Evaluation

The model can be evaluated using suitable classification metrics, including:

- **Accuracy:** Overall percentage of correct predictions.
- **Precision:** Correct positive predictions among all predicted positive cases.
- **Recall:** Positive cases correctly identified by the model.
- **F1-Score:** Balance between precision and recall.
- **Confusion Matrix:** Summary of correct and incorrect classifications.

Actual metric values should be added after testing the trained model on a held-out test dataset.

## 🔐 Input Validation and Reliability

- Reject empty or invalid numerical inputs.
- Check that the number and order of input features match the model's expected input.
- Apply the same preprocessing used during training.
- Handle prediction errors appropriately.
- Avoid claiming that arbitrary extreme values are safe merely because normalization is used.

## 🚀 Future Enhancements

- Add prediction history for users.
- Improve error messages and input validation.
- Display model evaluation charts.
- Add visual explanations of model predictions.
- Deploy the application to a cloud platform.
- Evaluate the model on independent datasets before considering any medical use.
