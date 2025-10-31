import torch
from datasets import Dataset
import math
import json
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./finetuned_model",
    max_seq_length=2048,
    load_in_4bit=True,
    device_map="auto"
)
model.eval()

# Replace "subfolder_name" with your actual folder path
with open("/home/incois/tvsubhaskar/llm_project/synthetic_data.json", "r") as f:
    synthetic_data = json.load(f)

formatted_data = [
    {"formatted_text": f"User: {item['input']}\nAssistant: {item['output']}"}
    for item in synthetic_data
]

synthetic_dataset = Dataset.from_list(formatted_data)

def compute_perplexity(example):
    inputs = tokenizer(
        example["formatted_text"],
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to("cuda")

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        perplexity = math.exp(loss.item()) if loss.item() < 100 else float("inf")
    return {"perplexity": perplexity}

synthetic_dataset = synthetic_dataset.map(compute_perplexity)
perplexities = synthetic_dataset["perplexity"]
avg_perplexity = sum(perplexities) / len(perplexities)

print(f"{avg_perplexity:.2f}")