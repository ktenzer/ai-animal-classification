import os
import json
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models

# === Paths ===
INFER_IMAGES_PATH = "./inference_images"
MODEL_CKPT_PATH = "finetuned-vision-model/webai/artifact/cls_models/best_weights.ckpt"
METADATA_PATH = "finetuned-vision-model/webai/artifact/metadata.json"

# === Load labels from metadata.json ===
with open(METADATA_PATH) as f:
    metadata = json.load(f)
    LABELS = metadata["labels"]  # now ['animal', 'not_animal']

# === Recreate the model architecture ===
def build_model(num_classes=2):
    base_model = ResNet50(include_top=False, weights="imagenet", pooling="avg", input_shape=(224, 224, 3))
    x = layers.Dense(128, activation="relu")(base_model.output)
    output = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs=base_model.input, outputs=output)

model = build_model(num_classes=len(LABELS))
model.load_weights(MODEL_CKPT_PATH)

# === Preprocessing function ===
def preprocess(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

# === Inference loop ===
image_files = [f for f in os.listdir(INFER_IMAGES_PATH) if f.lower().endswith((".jpg", ".png"))]
if not image_files:
    print("No images found in inference_images folder.")
    exit()

for img_name in image_files:
    img_path = os.path.join(INFER_IMAGES_PATH, img_name)
    image = Image.open(img_path).convert("RGB")
    input_tensor = preprocess(image)

    preds = model.predict(input_tensor)
    pred_label = LABELS[np.argmax(preds)]

    print(f"{img_name}: {pred_label}")