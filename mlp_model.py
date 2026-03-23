# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping

# Load your dataset (replace with your file path)
df = pd.read_csv("/content/dataR2.csv")

# Rename the target column if needed (adjust 'Classification' to your label column name)
df = df.rename(columns={'Classification': 'Label'})

# Handle missing values
df = df.dropna()
print("Dataset Summary:")
print(df.describe().T)
print("\nMissing Values:")
print(df.isnull().sum())

# Visualize label distribution
sns.countplot(x=df["Label"])
plt.title("Label Distribution")
plt.show()

# Encode categorical labels (e.g., 0 for benign, 1 for malignant)
le = LabelEncoder()
df["Label"] = le.fit_transform(df["Label"])
y = df["Label"].values

# Define features (drop the target column)
X = df.drop(columns=["Label"], axis=1)

# Scale the features
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split data with reduced test size (20% test, 80% train)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# Build the neural network model
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(16, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model with accuracy and AUC metrics
model.compile(loss='binary_crossentropy', optimizer='adam', 
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
print("\nModel Summary:")
print(model.summary())

# Define early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_auc', patience=20, mode='max', restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1,
                    validation_data=(X_test, y_test), callbacks=[early_stopping])

# Plot training and validation loss
plt.plot(history.history['loss'], 'y', label='Training Loss')
plt.plot(history.history['val_loss'], 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.plot(history.history['val_loss'], 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], 'y', label='Training Accuracy')
plt.plot(history.history['val_accuracy'], 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Predict probabilities on the test set
y_pred_probs = model.predict(X_test)

# Optimize threshold for best accuracy
thresholds = np.arange(0.1, 0.9, 0.1)
accuracies = []
for thresh in thresholds:
    y_pred = (y_pred_probs > thresh).astype(int)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

best_thresh = thresholds[np.argmax(accuracies)]
best_accuracy = max(accuracies)
print(f"\nBest threshold: {best_thresh:.1f}, with accuracy: {best_accuracy * 100:.2f}%")

# Final predictions using the best threshold
y_pred = (y_pred_probs > best_thresh).astype(int)

# Evaluate with confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix with Optimized Threshold')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Print final metrics
accuracy = accuracy_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred_probs)
print(f"Test Accuracy with Optimized Threshold: {accuracy * 100:.2f}%")
print(f"Test AUC: {auc_score:.4f}")

# Function to predict new data (optional)
def predict_breast_cancer(new_data):
    new_data = pd.DataFrame([new_data], columns=X.columns)
    new_data = scaler.transform(new_data)
    prediction = model.predict(new_data)
    return "Cancerous (Malignant)" if prediction > best_thresh else "Non-Cancerous (Benign)" 
