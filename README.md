# Breast Cancer Prediction System & Web App 🎗️

A full-stack computer-aided diagnosis (CAD) web application that leverages machine learning to improve the early detection and diagnosis of breast cancer. The system is designed to overcome the limitations of traditional analytical methods by effectively distinguishing between benign and malignant tumors through advanced feature extraction and classification.

## 🚀 Core Features
- **Advanced Feature Extraction:** Designed to leverage Convolutional Neural Networks (CNNs) to analyze mammogram data and extract high-dimensional features such as shape, texture, and density.
- **High-Accuracy Classification:** Implements Extreme Learning Machine (ELM) algorithms to efficiently and reliably categorize tumors[cite: 5].
- **Risk Ratio Analysis:** Outputs both a binary classification (Benign/Malignant) and a precise decimal probability percentage.
- **Strict Input Validation:** Form logic prevents empty submissions and ensures the model only processes valid, user-provided data.
- **Responsive UI:** A clean, dark-mode user interface styled with custom CSS and Bootstrap, designed to stretch dynamically and work perfectly on all screen sizes.
- **Robust Input Handling:** Includes data normalization layers to ensure extreme inputs do not break the prediction mathematical logic.

## 💻 Tech Stack & Requirements

### Software
- **Backend & ML:** Python, Flask, TensorFlow, Keras, NumPy[cite: 5]
- **Frontend:** HTML5, CSS3, JavaScript, Bootstrap[cite: 5]
- **IDE:** Visual Studio Code, Notepad[cite: 5]
- **Operating System:** Windows[cite: 5]

### Hardware Setup
- **Processor:** Dual core processor 2.6.0 GHZ[cite: 5]
- **RAM:** 4GB[cite: 5]
- **Storage:** 160 GB Hard disk[cite: 5]

## 🏗️ System Modules
- **Data Collection and Preprocessing:** Standardizes input dimensions and applies normalization and data augmentation techniques (e.g., rotation, flipping) to increase dataset diversity[cite: 5].
- **Machine Learning Model Development:** Utilizes convolutional, pooling, and fully connected layers to detect and analyze spatial features[cite: 5].
- **Training and Validation:** Minimizes prediction errors using optimized loss functions and validates accuracy on independent datasets[cite: 5].
- **Cancer Cell Detection Module:** Deploys model inference to evaluate unseen data and output probabilistic results[cite: 5].
- **Evaluation and Accuracy Testing:** Assesses system performance using metrics like accuracy, precision, recall, F1-score, and confusion matrices[cite: 5].
- **User Interface & Feedback:** Provides an accessible platform for uploading data, visualizing cancerous regions, and collecting user feedback to continuously retrain the model[cite: 5].

## 📊 Dataset
- **File:** `breast_cancer_data.csv`
- **Description:** Contains numerical measurements used as a structural reference and for testing real-world data distributions alongside primary diagnostic inputs.

## 📂 Project Structure
- `app.py`: The main Flask server application that handles web requests and interacts with the AI model.
- `train_model.py`: The script used to generate training data, build the neural network, and save the `.keras` model.
- `breast_cancer_model.keras`: The compiled and trained TensorFlow model.
- `breast_cancer_data.csv`: The dataset containing tumor measurement parameters.
- `static/index.html`: The frontend user interface containing the asynchronous fetch logic.

## 🛠️ How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/Breast-Cancer-Prediction-App.git](https://github.com/yourusername/Breast-Cancer-Prediction-App.git)
   cd Breast-Cancer-Prediction-App
