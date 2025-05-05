import gradio as gr
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification, CLIPProcessor, CLIPModel
import torch.nn.functional as F

# Preload models
TUNED_MODEL_PATH = "./finetuned-vision-model/model_torchscript.pt"
ORIGINAL_MODEL_NAME = "openai/clip-vit-base-patch32"

tuned_model = None
original_model = None
original_processor = None
custom_model = None
custom_processor = None

# Labels
LABELS = ["animal", "non_animal"]
ZERO_SHOT_LABELS = ["A photo of an animal", "A photo of a non-animal"]

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict(image, model_choice, custom_model_name):

    global tuned_model, original_model, original_processor, custom_model, custom_processor

    img = transform(image).unsqueeze(0)

    if model_choice == "Tuned":
        if tuned_model is None:
            tuned_model = torch.jit.load(TUNED_MODEL_PATH)
            tuned_model.eval()

        with torch.no_grad():
            logits = tuned_model(img)
            probs = F.softmax(logits, dim=-1)
            pred = logits.argmax(dim=-1).item()
            confidence = probs[0, pred].item()

        return f"Prediction: {LABELS[pred]} (Confidence: {confidence:.2f})"

    elif model_choice == "Original Zero Shot":
        if original_model is None:
            original_model = CLIPModel.from_pretrained(ORIGINAL_MODEL_NAME)
            original_processor = CLIPProcessor.from_pretrained(ORIGINAL_MODEL_NAME)
            original_model.eval()

        inputs = original_processor(text=ZERO_SHOT_LABELS, images=[image] * len(ZERO_SHOT_LABELS), return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            outputs = original_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            pred = probs[0].argmax(dim=-1).item()
            confidence = probs[0, pred].item()

        return f"Prediction: {ZERO_SHOT_LABELS[pred]} (Confidence: {confidence:.2f})"

    elif model_choice == "Custom":
        if not custom_model_name:
            return "Error: Please enter custom model name."

        if custom_model is None or custom_processor is None:
            custom_processor = AutoImageProcessor.from_pretrained(custom_model_name)
            custom_model = AutoModelForImageClassification.from_pretrained(custom_model_name)
            custom_model.eval()

        inputs = custom_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = custom_model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            pred = logits.argmax(dim=-1).item()
            confidence = probs[0, pred].item()

        labels = custom_model.config.id2label
        label = labels.get(pred, f"Label {pred}")

        return f"Prediction: {label} (Confidence: {confidence:.2f})"

    else:
        return "Invalid model selected."

# UI
with gr.Blocks() as iface:
    gr.Markdown("## Animal Detector - Multi Model\nUpload an image and choose model: Tuned, Original Zero Shot, or Custom model.")

    with gr.Row():
        model_choice = gr.Dropdown(choices=["Tuned", "Original Zero Shot", "Custom"], value="Tuned", label="Select Model")
        custom_model_name = gr.Textbox(label="Custom Model Name (HuggingFace)", visible=False)

    image_input = gr.Image(type="pil", label="Upload Image")
    output_text = gr.Textbox(label="Prediction")

    def update_custom_model_visibility(choice):
        return gr.update(visible=(choice == "Custom"))

    model_choice.change(update_custom_model_visibility, inputs=model_choice, outputs=custom_model_name)

    predict_btn = gr.Button("Predict")
    predict_btn.click(predict, inputs=[image_input, model_choice, custom_model_name], outputs=output_text)

iface.launch()