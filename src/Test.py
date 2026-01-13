import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.models import load_model
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Fix tqdm import for both Colab and local
try:
    from tqdm.notebook import tqdm
    print("Using notebook tqdm")
except ImportError:
    try:
        from tqdm import tqdm
        print("Using standard tqdm")
    except ImportError:
        # Fallback if tqdm is not installed
        print("tqdm not found, using basic progress")
        class tqdm:
            def __init__(self, total=0, desc="", unit=""):
                self.total = total
                self.n = 0
                self.desc = desc
            def update(self, n=1):
                self.n += n
                if self.n % 100 == 0 or self.n == self.total:
                    print(f"\r{self.desc}: {self.n}/{self.total}", end="")
            def __enter__(self):
                return self
            def __exit__(self, *args):
                print()


# ===========================
# STEP 1: Find the base path 
# ===========================
# Check if running in Google Colab
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    if os.path.exists('/content/drive/MyDrive'):
        print("✓ Google Drive already mounted")
        base_path = '/content/drive/MyDrive/Traffic_Sign_Recognition'
    else:
        try:
            from google.colab import drive
            drive.mount('/content/drive', force_remount=False)
            base_path = '/content/drive/MyDrive/Traffic_Sign_Recognition'
            print("✓ Google Drive mounted successfully")
        except:
            base_path = '/content/Traffic_Sign_Recognition'
            print("Using local Colab path")
else:
    # Local system paths
    possible_paths = [
        os.path.join(os.getcwd(), 'Traffic_Sign_Recognition'),
        os.getcwd(),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Traffic_Sign_Recognition'),
        os.path.dirname(os.path.abspath(__file__))
    ]
    
    base_path = None
    for path in possible_paths:
        if os.path.exists(path):
            # Check if Test.csv exists in this path
            if os.path.exists(os.path.join(path, 'Test.csv')):
                base_path = path
                break
    
    if base_path is None:
        base_path = os.getcwd()
    
    print(f"Running on local system")

print(f"Base path: {base_path}")

if not os.path.exists(base_path):
    print(f"ERROR: Base path {base_path} does not exist!")
    raise FileNotFoundError(f"Base path not found: {base_path}")

# ===============================
# STEP 2: Load the trained model
# ===============================
print("\nLoading trained model...")
model_locations = [
    os.path.join(base_path, 'my_model.h5'),
    os.path.join(os.getcwd(), 'my_model.h5'),
    'my_model.h5'
]

if IN_COLAB:
    model_locations.extend([
        '/content/drive/MyDrive/my_model.h5',
        '/content/my_model.h5',
        os.path.join(base_path, 'models', 'my_model.h5')
    ])

model = None
print("Searching for model in following locations:")
for model_path in model_locations:
    print(f"  Checking: {model_path}")
    if os.path.exists(model_path):
        model = load_model(model_path)
        print(f"✓ Model loaded from: {model_path}")
        break

if model is None:
    print("\n" + "="*60)
    print("ERROR: MODEL FILE NOT FOUND!")
    print("="*60)
    print("Please upload 'my_model.h5' to one of these locations:")
    for path in model_locations[:3]:  # Show only relevant paths
        print(f"  • {path}")
    if IN_COLAB:
        print("\n💡 TIP: Use this code to upload model:")
        print("from google.colab import files")
        print("uploaded = files.upload()")
    print("="*60)
    raise FileNotFoundError("Could not find my_model.h5 in any location!")

# =================================
# STEP 3: Load test data from CSV
# =================================
print("\nLoading test data from CSV...")
csv_path = os.path.join(base_path, 'Test.csv')

if not os.path.exists(csv_path):
    print(f"ERROR: Test.csv not found at {csv_path}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Files in base_path: {os.listdir(base_path) if os.path.exists(base_path) else 'Path does not exist'}")
    raise FileNotFoundError("Test.csv not found")

y_test_df = pd.read_csv(csv_path)
print(f"✓ CSV loaded with {len(y_test_df)} entries")

labels_csv = y_test_df["ClassId"].values
imgs = y_test_df["Path"].values
total_images = len(imgs)

print(f"✓ Sample path from CSV: {imgs[0]}")

# =======================
# STEP 4: IMAGE LOADING 
# =======================
print("\n" + "="*60)
print("     LOADING IMAGES...")
print("="*60)

def load_image(args):
    """Loads and processes a single image"""
    idx, img_path, label = args
    
    try:
        # Build full image path
        full_path = os.path.join(base_path, img_path)
        
        # Open image and resize to 30x30
        img = Image.open(full_path).resize((30, 30))
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Ensure RGB format (some images might be grayscale)
        if len(img_array.shape) == 2:  # Grayscale
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = img_array[:, :, :3]
        
        return idx, img_array, label, True
        
    except Exception as e:
        return idx, None, None, False


start_time = time.time()

# Create empty arrays for images and labels
total_images = len(imgs)
X_test_csv = np.zeros((total_images, 30, 30, 3), dtype=np.uint8)
valid_labels = np.zeros(total_images, dtype=np.int32)
valid_mask = np.zeros(total_images, dtype=bool)

# Determine number of parallel workers
num_workers = min(64, (os.cpu_count() or 4) * 8)
print(f"Workers: {num_workers}")

# Create task list: (index, path, label) for each image
tasks = [(i, img_path, label) for i, (img_path, label) in enumerate(zip(imgs, labels_csv))]

# Start parallel processing
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    with tqdm(total=total_images, desc="Loading", unit="img") as pbar:
        
        # Load each image in parallel
        for idx, img_array, label, success in executor.map(load_image, tasks):
            if success:
                X_test_csv[idx] = img_array
                valid_labels[idx] = label
                valid_mask[idx] = True
            pbar.update(1)

# Keep only successfully loaded images
X_test_csv = X_test_csv[valid_mask]
labels_csv = valid_labels[valid_mask]

# Normalize images to 0-1 range
print("Normalizing...")
X_test_csv = X_test_csv.astype(np.float32) / 255.0

# Show results
failed = total_images - np.sum(valid_mask)
loading_time = time.time() - start_time

print(f"\n✓ Loaded: {len(X_test_csv)} images")
print(f"✓ Time: {loading_time:.2f} seconds")
print(f"✓ Speed: {len(X_test_csv)/loading_time:.0f} images/sec")
if failed > 0:
    print(f"⚠ Failed: {failed} images")

if len(X_test_csv) == 0:
    raise ValueError("No images loaded!")

print(f"✓ Shape: {X_test_csv.shape}")


# ====================
# STEP 5: PREDICTIONS 
# ====================
print("\n" + "="*60)
print("      MAKING PREDICTIONS...")
print("="*60)

pred_start = time.time()

# Set batch size based on dataset size
if len(X_test_csv) > 10000:
    batch_size = 512
elif len(X_test_csv) > 5000:
    batch_size = 384
else:
    batch_size = 256

print(f"Batch size: {batch_size}")

# Get predictions from model
predictions_csv = model.predict(X_test_csv, batch_size=batch_size, verbose=1)

# Select class with highest probability
pred_csv = np.argmax(predictions_csv, axis=1)

pred_time = time.time() - pred_start
print(f"✓ Time: {pred_time:.2f} seconds")
print(f"✓ Speed: {len(X_test_csv)/pred_time:.0f} images/sec")

# ============================
# STEP 6: Calculate accuracy
# ============================
test_accuracy = accuracy_score(labels_csv, pred_csv)

print("\n" + "="*60)
print(f"        TEST ACCURACY: {test_accuracy * 100:.2f}%")
print("="*60)

# ==================================
# STEP 7: Generate confusion matrix
# ==================================
print("\nGenerating confusion matrix...")
cm_csv = confusion_matrix(labels_csv, pred_csv)

plt.figure(figsize=(15, 12))
sns.heatmap(cm_csv, annot=False, fmt='d', cmap='Greens', cbar=True)
plt.title('Confusion Matrix - Test Data', fontsize=16)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.savefig('confusion_matrix_test.png', dpi=300, bbox_inches='tight')
print("✓ Confusion matrix saved!")
plt.show()

# ===========================
# STEP 8: Sample predictions
# ===========================
print("\n" + "="*60)
print("         SAMPLE PREDICTIONS")
print("="*60)
print(f"{'#':<5} {'True':<10} {'Pred':<10} {'Status':<10}")
print("-"*60)

# Show first 15 predictions
sample_size = min(15, len(labels_csv))
for i in range(sample_size):
    status = "✓ CORRECT" if labels_csv[i] == pred_csv[i] else "✗ WRONG"
    print(f"{i+1:<5} {labels_csv[i]:<10} {pred_csv[i]:<10} {status:<10}")

# ====================
# STEP 9: Statistics
# ====================
# Vectorized operations for statistics
correct = np.sum(labels_csv == pred_csv)
total = len(labels_csv)
total_time = loading_time + pred_time

print("\n" + "="*60)
print("        PERFORMANCE STATISTICS")
print("="*60)
print(f"Total images: {total}")
print(f"Correct: {correct} ({correct/total*100:.2f}%)")
print(f"Incorrect: {total-correct} ({(total-correct)/total*100:.2f}%)")
print(f"\n⏱️ TOTAL TIME :")
print(f"  • Image loading: {loading_time:.2f}s ({len(X_test_csv)/loading_time:.0f} img/s)")
print(f"  • Predictions: {pred_time:.2f}s ({len(X_test_csv)/pred_time:.0f} img/s)")
print(f"  • Total time: {total_time:.2f}s")
print(f"\n⚡ Overall speed: {total/total_time:.0f} images/second")
print(f"\n🎯 Features used:")
print(f"  ✓ Parallel processing with {num_workers} workers")
print(f"  ✓ Boolean masking for fast filtering")
print(f"  ✓ Pre-allocated arrays for memory efficiency")
print(f"  ✓ Dynamic batch size optimization")
print("="*60)
print("\n✓    ⚡ TESTING COMPLETE! ⚡")
print("="*60)

# Keep window open on local system
if not IN_COLAB:
    print("\n" + "="*60)
    print("Press ENTER to exit...")
    print("="*60)
    input()