import os
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoImageProcessor, CLIPForImageClassification, Trainer, TrainingArguments, AutoConfig
from datasets import Dataset, DatasetDict
from roboflow import Roboflow
from collections import Counter
from torch import nn

# --- Configuration ---
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE")
ROBOFLOW_PROJECT = os.getenv("ROBOFLOW_PROJECT")
ROBOFLOW_VERSION = "6"

MODEL_NAME = "openai/clip-vit-base-patch32"
LABEL_MAP = {"animal": 0, "not_animal": 1}
BATCH_SIZE = 12
EPOCHS = 3
OUTPUT_DIR = "./finetuned-vision-model"

# Download dataset (Roboflow)
#print("Downloading dataset from Roboflow...")
#rf = Roboflow(api_key=ROBOFLOW_API_KEY)
#project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
#dataset = project.version(ROBOFLOW_VERSION).download("folder")
#dataset_path = dataset.location

# --- Dataset ---
dataset_path = "./Animal-Classification-6"

def load_data(data_dir, label_map):
    data = []
    for label_name in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label_name)
        if not os.path.isdir(label_dir):
            continue
        label = label_map.get(label_name.lower())
        if label is None:
            continue
        for fname in os.listdir(label_dir):
            if fname.endswith('.jpg') or fname.endswith('.png'):
                data.append({
                    "image_path": os.path.join(label_dir, fname),
                    "label": label
                })
    return data

train_data = load_data(os.path.join(dataset_path, "train"), LABEL_MAP)
val_data = load_data(os.path.join(dataset_path, "test"), LABEL_MAP)

print("Train Samples:", len(train_data))
print("Validation Samples:", len(val_data))

# --- Prepare datasets ---
def preprocess(example):
    img = Image.open(example["image_path"]).convert("RGB")
    example["image"] = img
    return example

train_dataset = Dataset.from_list(train_data).map(preprocess)
val_dataset = Dataset.from_list(val_data).map(preprocess)

datasets = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset
})

# Calculate weights automatically
# labels = [x["label"] for x in datasets["train"]]
# counter = Counter(labels)
# total = sum(counter.values())
# class_weights = [total / counter[i] for i in range(len(counter))]
# class_weights = torch.tensor(class_weights, dtype=torch.float)

# Calculate weights manually
class_weights = torch.tensor([6.1, 2.1]) 

print("Using class weights:", class_weights)

# Custom model with weighted loss and label smoothing
class WeightedLossModel(CLIPForImageClassification):
    def __init__(self, config, class_weights=None, label_smoothing=0.1):
        super().__init__(config)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    def forward(self, pixel_values=None, labels=None, **kwargs):
        outputs = super().forward(pixel_values=pixel_values, labels=None, **kwargs)
        logits = outputs.logits
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}

# Load pretrained config and initialize custom model
config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=len(LABEL_MAP))
model = WeightedLossModel(config=config, class_weights=class_weights, label_smoothing=0.1)

# Load pretrained weights to avoid cumulative drift
pretrained_state_dict = CLIPForImageClassification.from_pretrained(MODEL_NAME).state_dict()
model.load_state_dict(pretrained_state_dict, strict=False)

# Image transforms 
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

augment = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])

def transform_train(example):
    example["image"] = augment(example["image"])
    encoding = processor(images=example["image"], return_tensors="pt")
    example["pixel_values"] = encoding["pixel_values"][0]
    example["labels"] = torch.tensor(example["label"])
    return example

def transform_val(example):
    encoding = processor(images=example["image"], return_tensors="pt")
    example["pixel_values"] = encoding["pixel_values"][0]
    example["labels"] = torch.tensor(example["label"])
    return example

datasets["train"] = datasets["train"].map(transform_train)
datasets["validation"] = datasets["validation"].map(transform_val)

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=EPOCHS,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir="./logs",
    save_total_limit=1,
    resume_from_checkpoint=False
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    acc = (predictions == labels).mean()
    return {"accuracy": acc}

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=datasets["train"],
    eval_dataset=datasets["validation"],
    compute_metrics=compute_metrics
)

# Train
print("Starting training...")
trainer.train()

# Evaluate
results = trainer.evaluate()
print("Final accuracy:", results)

# Save model
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

# Export to TorchScript
print("Exporting to TorchScript...")

class WrappedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        return outputs["logits"]

wrapped_model = WrappedModel(model.cpu())
wrapped_model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
traced_model = torch.jit.trace(wrapped_model, dummy_input)
traced_model.save(os.path.join(OUTPUT_DIR, "model_torchscript.pt"))

print("TorchScript export complete!")
print("Done! Model saved to", OUTPUT_DIR)
