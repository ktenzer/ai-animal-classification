import gradio as gr
import torch
from PIL import Image
from torchvision import transforms

# Load model
MODEL_PATH = "./finetuned-vision-model/model_torchscript.pt"
model = torch.jit.load(MODEL_PATH)
model.eval()

# Labels
LABELS = ["animal", "non_animal"]

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Prediction function
def predict(image):
    img = transform(image).unsqueeze(0) 

    with torch.no_grad():
        logits = model(img)
        pred = logits.argmax(dim=-1).item()

    return f"Prediction: {LABELS[pred]}"

# UI
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Animal Detector",
    description="Upload an image to check if it's an animal or not."
)

iface.launch()