# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

# Initialize empty lists for storing data and labels
data = []
labels = []
classes = 43  # Total number of traffic sign classes

# ============================
#  Mount Google Drive first.
# ============================
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("✓ Google Drive mounted successfully")

    # Try Google Drive paths
    possible_paths = [
        '/content/drive/MyDrive/Traffic_Sign_Recognition/Train',
        '/content/drive/My Drive/Traffic_Sign_Recognition/Train',
    ]
except:
    possible_paths = []
    print("Not running in Colab or Drive already mounted")

# ==================================================
# Add local paths (if dataset is uploaded to Colab)
# ==================================================
possible_paths.extend([
    '/content/Traffic_Sign_Recognition/Train',
    'Traffic_Sign_Recognition/Train',
    'Train',
])

print("\nSearching for training data...")
print("Checking these locations:")
for p in possible_paths:
    print(f"  - {p}")

# Find the correct base path
base_path = None
for path in possible_paths:
    if os.path.exists(path):
        # Verify it has numbered subdirectories
        if os.path.exists(os.path.join(path, '0')) or os.path.exists(os.path.join(path, '1')):
            base_path = path
            print(f"\n✓ Found training data at: {base_path}")
            break
if base_path is None:
    print("ERROR: Could not find training data folder!")
    raise FileNotFoundError("Training data folder not found")


print("\nLoading training images...")

# ============================================
# STEP 1: Load and preprocess training images
# ============================================
loaded_classes = 0
for i in range(classes):
    path = os.path.join(base_path, str(i))

    # Check if folder exists
    if not os.path.exists(path):
        print(f"⚠ Warning: Class {i} folder not found at {path}")
        continue

    images = os.listdir(path)
    if len(images) == 0:
        print(f"⚠ Warning: Class {i} folder is empty")
        continue

    loaded_classes += 1
    print(f"✓ Loading class {i:2d}: {len(images):4d} images", end='\r')

    for a in images:
        try:
            # Open image and resize to 30x30 pixels
            image = Image.open(os.path.join(path, a))
            image = image.resize((30, 30))
            image = np.array(image)

            # Append to data and labels
            data.append(image)
            labels.append(i)
        except Exception as e:
            print(f"\n⚠ Error loading {a}: {e}")

print(f"\n✓ Successfully loaded {loaded_classes} classes")

# Convert lists to numpy arrays
data = np.array(data)
labels = np.array(labels)

print(f"\n{'='*31}")
print(f"Data shape: {data.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Total images loaded: {len(data)}")
print(f"{'='*31}\n")

if len(data) == 0:
    raise ValueError("No images were loaded! Please check your data directory structure.")

# ===============================================
# STEP 2: Normalize pixel values to range [0, 1]
# ===============================================
data = data / 255.0

# ==================================================
# STEP 3: Split data into training and testing sets
# ==================================================
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")

# ===================================
# STEP 4: Convert labels to encoding
# ===================================
y_test_original = y_test.copy()
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

print("\nBuilding CNN model...")

# =========================
# STEP 5: Build CNN Model
# =========================
model = Sequential()

# First Convolutional Block
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu',
                 input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))

# Second Convolutional Block
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))

# Fully Connected Layers
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))

# ==========================
# STEP 6: Compile the model
# ==========================
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print(model.summary())

print("\nTraining model...")

# =========================
# STEP 7: Train the model
# =========================
epochs = 15
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=epochs,
    validation_data=(X_test, y_test)
)

print("\nSaving model...")
model.save("my_model.h5")

# =============================
# STEP 8: Plot accuracy graph
# =============================
print("Plotting accuracy graph...")
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# =========================
# STEP 9: Plot loss graph
# =========================
print("Plotting loss graph...")
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# ===================================
# STEP 10: Generate confusion matrix
# ===================================
print("Generating confusion matrix for validation data...")

val_predictions = model.predict(X_test)
val_pred_classes = np.argmax(val_predictions, axis=1)

cm = confusion_matrix(y_test_original, val_pred_classes)

plt.figure(figsize=(15, 12))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', cbar=True)
plt.title('Confusion Matrix - Validation Data', fontsize=16)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.savefig('confusion_matrix_validation.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*25)
print("   TRAINING COMPLETE!")
print("="*60)
print(f"✓ Model saved as: my_model.h5")
print(f"✓ Accuracy plot saved")
print(f"✓ Loss plot saved")
print(f"✓ Confusion matrix saved")
print("="*25)
