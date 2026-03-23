# Step 1: Install Kaggle API
!pip install kaggle

# Step 2: Upload Kaggle API Key
from google.colab import files
files.upload()  # Upload kaggle.json (download it from your Kaggle account settings)

# Step 3: Move kaggle.json to the right directory
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Step 4: Download dataset
!kaggle datasets download -d paultimothymooney/breast-histopathology-images

# Step 5: Extract dataset
import zipfile
import os

dataset_path = "breast-histopathology-images.zip"
extract_path = "./breast-histopathology-images"

with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print(f"Dataset extracted to: {extract_path}")    


import os
import random
import glob
import tensorflow as tf

# Constants
IMG_SIZE = 96
BATCH_SIZE = 4
EPOCHS = 10  # Reduced to 10 to prevent overfitting

# Dataset path
dataset_path = "/content/breast-histopathology-images"
breast_img_paths = glob.glob(os.path.join(dataset_path, "", "*.png"), recursive=True)
print(f"Total images found: {len(breast_img_paths)}")

# Separate classes
class_0_paths = [path for path in breast_img_paths if os.path.basename(os.path.dirname(path)) == '0']
class_1_paths = [path for path in breast_img_paths if os.path.basename(os.path.dirname(path)) == '1']
print(f"Class 0 images: {len(class_0_paths)}")
print(f"Class 1 images: {len(class_1_paths)}")

# Check if there are enough images for the new split
if len(class_0_paths) < 105000 or len(class_1_paths) < 105000:
    raise ValueError("Not enough images to assign 75,000 per class for training, 5,000 for validation, and 25,000 for testing.")

# Shuffle and split
random.shuffle(class_0_paths)
random.shuffle(class_1_paths)
train_paths = class_0_paths[:75000] + class_1_paths[:75000]  # 150,000 total training
images
random.shuffle(train_paths)
val_paths = class_0_paths[75000:80000] + class_1_paths[75000:80000]  # 10,000 total validation images
test_paths = class_0_paths[80000:105000] + class_1_paths[80000:105000]  # 50,000 total testing images

# Data augmentation to reduce overfitting
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
])

# Process image function
def process_path(file_path):
    label = tf.where(tf.strings.regex_full_match(file_path, ".[\\/](1)[\\/]."), 1, 0)
    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = img / 255.0
    return img, label

# Datasets with augmentation for training
train_ds = tf.data.Dataset.from_tensor_slices(train_paths).map(
    process_path, num_parallel_calls=tf.data.AUTOTUNE
).map(
    lambda img, label: (data_augmentation(img, training=True), label), num_parallel_calls=tf.data.AUTOTUNE
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices(val_paths).map(
process_path, num_parallel_calls=tf.data.AUTOTUNE
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices(test_paths).map(
    process_path, num_parallel_calls=tf.data.AUTOTUNE
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Model with dropout to reduce overfitting
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),  # Added dropout to prevent overfitting
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Learning rate schedule
initial_learning_rate = 0.0001  # Lowered to stabilize training
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=10000, decay_rate=0.9, staircase=True)

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC()]  # Added AUC to monitor performance
)

# Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# Evaluate on test set
test_loss, test_accuracy, test_auc = model.evaluate(test_ds)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
print(f"Test AUC: {test_auc}")

import tensorflow as tf
import matplotlib.pyplot as plt
import os

# Define the image size (assuming the model was trained with 96x96 images)
IMG_SIZE = 96

# Load the trained model from Google Drive
model_path = '/content/drive/MyDrive/mavyan.h5'
try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Function to classify and display a single image
def classify_and_display_image(image_path):
    try:
        # Load image for display
        img = tf.io.read_file(image_path)
        img = tf.image.decode_image(img, channels=3)  # Supports multiple image formats
        img_display = img.numpy()  # Convert to numpy array for display

        # Preprocess image for prediction (resize and normalize)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = img / 255.0  # Normalize to [0, 1]
        img = tf.expand_dims(img, 0)  # Add batch dimension

        # Make prediction
        prob = model.predict(img)[0][0]  # Get probability from sigmoid output
        prediction = "Cancer" if prob > 0.5 else "No Cancer"
confidence = prob if prob > 0.5 else 1 - prob  # Confidence in the predicted class

        # Display the image with prediction
        plt.figure(figsize=(5, 5))
        plt.imshow(img_display)
        plt.title(f"Prediction: {prediction}\nConfidence: {confidence*100:.2f}%")
        plt.axis('off')
        plt.show()

        # Print the result to console
        print(f"Prediction: {prediction}, Confidence: {confidence*100:.2f}%")
    except Exception as e:
        print(f"Error processing image: {e}")

# Interactive testing loop
print("Enter the image path to classify, or type 'exit' to stop.")
while True:
    image_path = input("Image path: ")
    if image_path.lower() == 'exit':
        break
    if not os.path.exists(image_path):
        print("Error: Image path does not exist. Please try again.")
        continue
    classify_and_display_image(image_path)

print("Testing complete.")

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import seaborn as sns
import glob
import os

# Constants (adjust these if different in your training setup)
IMG_SIZE = 96  # Image size used during training (e.g., 96x96)
BATCH_SIZE = 4  # Batch size used during training

# Function to process image paths (same as used in training)
def process_path(file_path):
    # Extract label from file path (assuming '0' or '1' in directory name)
    label = tf.where(tf.strings.regex_full_match(file_path, ".[\\/](1)[\\/]."), 1, 0)
img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels=3)  # Assuming PNG images with 3 channels
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = img / 255.0  # Normalize to [0, 1]
    return img, label

# Define test data paths (adjust this based on your dataset location)
dataset_path = "/content/breast-histopathology-images"  # Example path, update as needed
breast_img_paths = glob.glob(os.path.join(dataset_path, "", "*.png"), recursive=True)
class_0_paths = [path for path in breast_img_paths if os.path.basename(os.path.dirname(path)) == '0']
class_1_paths = [path for path in breast_img_paths if os.path.basename(os.path.dirname(path)) == '1']
test_paths = class_0_paths[17500:21500] + class_1_paths[17500:21500]  # Example split, adjust as needed

# Create test dataset
test_ds = tf.data.Dataset.from_tensor_slices(test_paths).map(
    process_path, num_parallel_calls=tf.data.AUTOTUNE
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Load the trained model
model = tf.keras.models.load_model('/content/mavyan.h5')

# Make predictions on the test dataset
test_predictions = model.predict(test_ds)
pred_probs = test_predictions.flatten()  # Predicted probabilities
pred_labels = (pred_probs > 0.5).astype(int)  # Predicted labels (threshold = 0.5)

# Extract true labels from the test dataset
true_labels = []
for _, labels in test_ds:
    true_labels.extend(labels.numpy())
true_labels = np.array(true_labels)

# --- Plot 1: Confusion Matrix ---
cm = confusion_matrix(true_labels, pred_labels)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['No Cancer', 'Cancer'], yticklabels=['No Cancer', 'Cancer'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# --- Plot 2: ROC Curve ---
fpr, tpr, _ = roc_curve(true_labels, pred_probs)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# --- Plot 3: Precision-Recall Curve ---
precision, recall, _ = precision_recall_curve(true_labels, pred_probs)
pr_auc = auc(recall, precision)
print(f"Precision-Recall AUC: {pr_auc:.4f}")

plt.figure()
plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()

# Evaluate the model on the test set
test_loss, test_accuracy, test_auc = model.evaluate(test_ds)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test AUC: {test_auc:.4f}")































