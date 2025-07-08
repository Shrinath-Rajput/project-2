# project-2
Automated Waste Segregation System Using Opencv and Image Processing 
This project is a complete system that detects and classifies different types of garbage using image processing techniques and stores the data into a MySQL database for later analysis and retrieval.

## 🔧 Features
- Real-time garbage image input using camera or manual upload
- Image classification into types like:
  - Wet Waste
  - Dry Waste
  - Plastic
  - Metal
  - Rubber
- Data storage in MySQL (image path and type)
- Retrieval and display of stored classification history

## 🛠️ Tech Stack
- Python
- OpenCV
- TensorFlow / Keras (for model)
- MySQL
- Google Colab (for development & testing)

## 💡 Use Cases
- Smart Dustbins
- Waste Management Systems
- AI in Municipal Applications

## 📂 How to Run
1. Clone the repository
2. Install requirements
3. Connect your MySQL database
4. Run the main Python script to start detecting and classifying garbage.

## 📎 Colab Notebook
Access the notebook here:  
[Google Colab Link](https://colab.research.google.com/drive/1DrIIxyerGB05QCJTGck3U66FM3Li-r-z)

## 📸 Sample Output
![output example](add-sample-output-image-if-you-have)

## 📧 Contact
Feel free to connect with me on [LinkedIn](your-profile-link) for collaboration or suggestions.
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2, VGG16, ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

# --- CONFIG ---
IMG_SIZE = 224
EPOCHS = 5
BATCH_SIZE = 16
DATASET_DIR = "/content/drive/MyDrive/dataset/garbageprojectimages/garbage-dataset"
MODELS = {
    "MobileNetV2": (MobileNetV2, mobilenet_preprocess),
    "VGG16": (VGG16, vgg_preprocess),
    "ResNet50": (ResNet50, resnet_preprocess),
}

# --- Load images from folders ---
X, y, labels = [], [], []
label_map = {}
label_counter = 0

for folder in sorted(os.listdir(DATASET_DIR)):
    path = os.path.join(DATASET_DIR, folder)
    if not os.path.isdir(path): continue # Ensure it's a directory

    if folder not in label_map:
        label_map[folder] = label_counter
        label_counter += 1

    for file in os.listdir(path):
        img_path = os.path.join(path, file)
        # Check if the item in the folder is a file before trying to load it
        if os.path.isfile(img_path):
            try:
                img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
                img_array = img_to_array(img)
                X.append(img_array)
                y.append(label_map[folder])
            except:
                continue

X = np.array(X)
y = np.array(y)
CATEGORIES = list(label_map.keys())
y_cat = to_categorical(y, num_classes=len(CATEGORIES))

# --- Train/Test Split ---
# Adjusted train_size to 0.8 to ensure a non-empty training set
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42,train_size=0.8)

# --- Train & Evaluate Models ---
results = {}

for model_name, (model_class, preprocess) in MODELS.items():
    print(f"\n🔧 Training {model_name}...")

    X_train_p = preprocess(X_train.copy().astype('float32'))
    X_test_p = preprocess(X_test.copy().astype('float32'))

    base_model = model_class(weights="imagenet", include_top=False, input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3)))
    for layer in base_model.layers:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(len(CATEGORIES), activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train_p, y_train, validation_data=(X_test_p, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE)

    y_pred_probs = model.predict(X_test_p)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    label=[0,1,2,3,4,5,6,7,8,9]

    report = classification_report(y_true, y_pred,labels=label, target_names=CATEGORIES, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    acc = report['accuracy']
    precision = np.mean([report[label]['precision'] for label in CATEGORIES])
    recall = np.mean([report[label]['recall'] for label in CATEGORIES])
    f1 = np.mean([report[label]['f1-score'] for label in CATEGORIES])

    # --- Store results ---
    results[model_name] = {
        "accuracy": acc * 100,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "report": report,
        "conf_matrix": cm,
        "history": history,
        "y_true": y_true,
        "y_pred_probs": y_pred_probs
    }

    # --- Plot Accuracy & Loss ---
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'{model_name} - Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{model_name} - Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Confusion Matrix ---
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CATEGORIES, yticklabels=CATEGORIES)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()

    # --- ROC Curve (macro-average)
    plt.figure()
    fpr = dict()
    tpr = dict()
    for i in range(len(CATEGORIES)):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_probs[:, i])
        plt.plot(fpr[i], tpr[i], label=f'{CATEGORIES[i]}')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'{model_name} - ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- Best Model ---
best_model = max(results, key=lambda k: results[k]['accuracy'])
print(f"\n🏆 Best Model: {best_model} with Accuracy = {results[best_model]['accuracy']:.2f}%")
