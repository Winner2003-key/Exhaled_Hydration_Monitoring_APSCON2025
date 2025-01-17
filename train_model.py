import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# Base directory containing the BPM folders
base_dir = "/content/dataset/Exhaled_hydration/Exhaled hydration - Augmented Dataset"
# Define thresholds for hydration levels
LOW_THRESHOLD = -30
HIGH_THRESHOLD = -10

# Initialize an empty list to store data
combined_data = []

# Iterate over the BPM folders
for bpm_folder in ["12BPM Exhaled Hydration", "20BPM Exhaled Hydration", "28BPM Exhaled Hydration"]:
    bpm_value = int(bpm_folder.split("BPM")[0])  # Extract BPM value from folder name
    folder_path = os.path.join(base_dir, bpm_folder)
    
    # Iterate over the patient files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            # Extract patient ID (e.g., P1) from the filename
            patient_id = filename.split("-")[1].strip()
            
            # Load the CSV file
            file_path = os.path.join(folder_path, filename)
            signal_data = pd.read_csv(file_path, header=None, names=['SignalDb'])
            
            # Add data to the combined list
            combined_data.append({
                "PatientID": patient_id,
                "BPM": bpm_value,
                "SignalDb": signal_data['SignalDb'].tolist()  # Convert the column to an array
            })

# Convert the combined list to a DataFrame
combined_df = pd.DataFrame(combined_data)

# Split the dataset into training and testing sets before further processing
train_df, test_df = train_test_split(combined_df, test_size=0.2, random_state=42)


def categorize_hydration(signal):
    """Categorize hydration level based on thresholds."""
    avg_signal = np.mean(signal)  # Use average signal value for classification
    if avg_signal < LOW_THRESHOLD:
        return "low"
    elif avg_signal > HIGH_THRESHOLD:
        return "high"
    else:
        return "normal"
    

# Apply categorization on training and testing datasets
train_df["HydrationLevel"] = train_df["SignalDb"].apply(categorize_hydration)
test_df["HydrationLevel"] = test_df["SignalDb"].apply(categorize_hydration)

# Prepare data for model training
X_train = np.array(train_df["SignalDb"].tolist())  # Features (signals)
y_train = train_df["HydrationLevel"]  # Labels for training

X_test = np.array(test_df["SignalDb"].tolist())  # Features (signals)
y_test = test_df["HydrationLevel"]  # Labels for testing

# Encode labels to integers
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),  # Input layer matching signal length
    tf.keras.layers.Dense(32, activation="relu"),  # Hidden layer
    tf.keras.layers.Dense(16, activation="relu"),  # Hidden layer
    tf.keras.layers.Dense(3, activation="softmax"),  # Output layer (3 classes: low, normal, high)
])

# Compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(X_train, y_train_encoded, epochs=50, validation_split=0.2, verbose=1)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test_encoded, verbose=0)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")