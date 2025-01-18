import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib
from sklearn.model_selection import GridSearchCV
import cv2
import seaborn as sns
import time
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Define paths
dataset_dir = r"C:\Study\Infotech\Task 3"
train_dir = os.path.join(dataset_dir, "train")
test_dir = os.path.join(dataset_dir, "test1")

# Ensure output directory exists
output_dir = os.path.join(dataset_dir, "Output")
os.makedirs(output_dir, exist_ok=True)

# Paths for outputs
confusion_image_path = os.path.join(output_dir, 'confusion_matrix.png')
classification_file_path = os.path.join(output_dir, 'classification_report.txt')
model_file_path = os.path.join(output_dir, "svm_model.pkl")

# Load and preprocess training images
train_images = [f for f in os.listdir(train_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
features = []
labels = []
image_size = (50, 50)

print("Processing train images...")
for image in tqdm(train_images, desc="Processing Train Images"):
    image_path = os.path.join(train_dir, image)
    label = 0 if image.startswith('cat') else 1  # Label: cat=0, dog=1
    image_read = cv2.imread(image_path)

    # Check if the image was read successfully
    if image_read is None:
        print(f"Warning: Unable to read image {image_path}. Skipping...")
        continue

    try:
        image_resized = cv2.resize(image_read, image_size)
    except Exception as e:
        print(f"Error resizing image {image_path}: {e}. Skipping...")
        continue

    image_normalized = image_resized / 255.0
    image_flatten = image_normalized.flatten()
    features.append(image_flatten)
    labels.append(label)

# Check if any images were successfully processed
if not features:
    print("Error: No valid images found in the training directory.")
    exit()

features = np.asarray(features)
labels = np.asarray(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=True, random_state=42)

# PCA, SVM, and Pipeline
n_components = 0.8
pca = PCA(n_components=n_components)
svm = SVC()
pipeline = Pipeline([
    ('pca', pca),
    ('svm', svm)
])

# Hyperparameter grid
param_grid = {
    'pca__n_components': [2, 1, 0.9, 0.8],
    'svm__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
}

# Train model using GridSearchCV
start_time = time.time()
grid_search = GridSearchCV(pipeline, param_grid, cv=3, verbose=4)
grid_search.fit(X_train, y_train)
end_time = time.time()

# Get best model and parameters
best_pipeline = grid_search.best_estimator_
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters: ", best_params)
print("Best Score: ", best_score)

# Evaluation on test dataset
accuracy = best_pipeline.score(X_test, y_test)
print("Accuracy:", accuracy)

y_pred = best_pipeline.predict(X_test)

# Classification report
target_names = ['Cat', 'Dog']
classification_rep = classification_report(y_test, y_pred, target_names=target_names)
print("Classification Report:\n", classification_rep)

with open(classification_file_path, 'w') as file:
    file.write(classification_rep)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.savefig(confusion_image_path)
plt.show()

# Save the model
joblib.dump(best_pipeline, model_file_path)
print(f"Model saved at {model_file_path}")
