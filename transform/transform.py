import io
import os
from contextlib import redirect_stderr, redirect_stdout

import torch
from transformers import logging as transformers_logging
from transformers import (
    AutoModelForSequenceClassification,
    BatchEncoding,
    BertTokenizer,
)

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
auth_kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}
transformers_logging.set_verbosity_error()

with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, **auth_kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        **auth_kwargs,
    ).to(DEVICE)

text = "这个产品很好用，我很满意。"
encoded_inputs: BatchEncoding = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding=True,
)
model_inputs: dict[str, torch.Tensor] = {
    key: value.to(DEVICE) for key, value in encoded_inputs.items()
}

with torch.no_grad():
    outputs = model(**model_inputs)

predicted_label_id = outputs.logits.argmax(dim=-1).item()
print(f"device: {DEVICE_NAME}")
print(f"text: {text}")
print(f"predicted label: {LABELS[predicted_label_id]}")
print("warning: bert-base-chinese is a base model, not a fine-tuned sentiment classifier.")
