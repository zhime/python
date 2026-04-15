import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "bert-base-chinese"
LABELS = ["negative", "positive"]
HF_TOKEN = os.getenv("HF_TOKEN")


def select_device() -> tuple[object, str]:
    if torch.cuda.is_available():
        return torch.device("cuda"), f"CUDA: {torch.cuda.get_device_name(0)}"

    try:
        import torch_directml

        return torch_directml.device(), "DirectML GPU"
    except ImportError:
        return torch.device("cpu"), "CPU"


DEVICE, DEVICE_NAME = select_device()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS),
    token=HF_TOKEN,
).to(DEVICE)

text = "这个产品很好用，我很满意。"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)

predicted_label_id = outputs.logits.argmax(dim=-1).item()
print(f"device: {DEVICE_NAME}")
print(f"text: {text}")
print(f"predicted label: {LABELS[predicted_label_id]}")
print("warning: bert-base-chinese is a base model, not a fine-tuned sentiment classifier.")
