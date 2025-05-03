import os
import argparse
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor, 
    AutoModelForImageClassification, 
    CLIPProcessor, 
    CLIPModel
)

# Settings
TUNED_MODEL_PATH = "./finetuned-vision-model/model_torchscript.pt"
TUNED_PROCESSOR_PATH = "./finetuned-vision-model"
ORIGINAL_MODEL_NAME = "openai/clip-vit-base-patch32"
INFER_IMAGES_PATH = "./inference_images"
ZERO_SHOT_LABELS = ["A photo of an animal", "A photo of a non-animal"]

# Args
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, choices=["tuned", "original", "original_zero_shot"], required=True, help="Model to use")
args = parser.parse_args()

# Load Images
image_files = [f for f in os.listdir(INFER_IMAGES_PATH) if f.endswith(('.jpg', '.png'))]

if len(image_files) == 0:
    print("Error: No images found in inference_images folder.")
    exit(0)

# Load Model
if args.model == "tuned":
    print("Loading tuned model...")
    model = torch.jit.load(TUNED_MODEL_PATH)
    model.eval()
    processor = AutoImageProcessor.from_pretrained(TUNED_PROCESSOR_PATH)

    LABELS = ["animal", "not_animal"]

    def predict(image):
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            logits = model(inputs["pixel_values"])
            pred = logits.argmax(dim=-1).item()
        return LABELS[pred]

elif args.model == "original_zero_shot":
    print("Loading original model in zero-shot mode...")
    model = CLIPModel.from_pretrained(ORIGINAL_MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(ORIGINAL_MODEL_NAME)
    model.eval()

    def predict(image):
        inputs = processor(text=ZERO_SHOT_LABELS, images=[image] * len(ZERO_SHOT_LABELS), return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

        pred = probs[0].argmax(dim=-1).item()
        return ZERO_SHOT_LABELS[pred]

elif args.model == "original":
    print("Loading original pretrained model...")
    model = AutoModelForImageClassification.from_pretrained(ORIGINAL_MODEL_NAME)
    model.eval()
    processor = AutoImageProcessor.from_pretrained(ORIGINAL_MODEL_NAME)

    LABELS = model.config.id2label  # e.g. {0: "LABEL_0", 1: "LABEL_1", ...}

    def predict(image):
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            pred = logits.argmax(dim=-1).item()
        return LABELS[pred]

else:
    raise ValueError("Error: Unknown model option selected.")

# --- INFERENCE LOOP ---
print("Running inference...")
for img_name in image_files:
    img_path = os.path.join(INFER_IMAGES_PATH, img_name)
    image = Image.open(img_path).convert("RGB")

    result = predict(image)
    print(f"{img_name}: {result}")

print("Inference complete.")