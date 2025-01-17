# Exhaled_Hydration_Monitoring_APSCON2025
# Exhaled Hydration Signal Classification
This project uses machine learning to classify hydration levels based on exhaled signal data. The dataset consists of exhaled signal measurements at different BPM levels, which are processed and used to train a neural network model to predict hydration levels.

## Dataset
The dataset contains exhaled signal data stored in CSV files, categorized into three BPM levels: 12 BPM, 20 BPM, and 28 BPM. Each CSV file contains signal data for different patients.

Path to dataset: /content/dataset/Exhaled_hydration/Exhaled hydration - Augmented Dataset
File format: CSV
Signal Data: The CSV files contain a column named SignalDb representing the exhaled signal.
Data Preprocessing
The preprocessing steps include:

Loading Data: Signal data from CSV files is loaded into a DataFrame.
Patient ID Extraction: Each file contains the patient ID, which is extracted from the filename.
Categorization: Hydration levels are categorized based on the average signal value:
Low: Average signal < -30
Normal: -30 <= Average signal <= 0
High: Average signal > 0
Dataset Split: The data is split into training (80%) and testing (20%) sets.
Model
The model is a neural network built using TensorFlow/Keras:

Input layer: Accepts signal data with the shape (signal_length,)
Hidden layers: Two dense layers with ReLU activation
Output layer: A dense layer with 3 outputs (low, normal, high) using softmax activation
Training:
Optimizer: Adam
Loss function: Sparse categorical cross-entropy
Epochs: 50
Validation split: 20%
Evaluation:
After training, the model is evaluated on the test set to determine its accuracy.

## Requirements
Python 3.x
TensorFlow
NumPy
Pandas
Scikit-learn
You can install the necessary dependencies using pip:

bash
Copy
Edit
pip install tensorflow numpy pandas scikit-learn
Usage
Clone or download the repository.
Place the dataset in the specified directory or modify the path in the script.
Run the script to preprocess the data, train the model, and evaluate its performance.
bash
Copy
Edit
python train_model.py
Results
After training, the model will output the test accuracy:

mathematica
Copy
Edit
Test Accuracy: XX.XX%
