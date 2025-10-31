import torch
from datasets import Dataset
import math
import json
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./llama3_oceanbench_lora",
    max_seq_length=2048,
    load_in_4bit=True,
    device_map="auto"
)
model.eval()

with open("/home/incois/tvsubhaskar/llm_project/synthetic_data.json", "r") as f:
    synthetic_data = json.load(f)

formatted_data = [
    {"formatted_text": f"User: {item['input']}\nAssistant: {item['output']}"}
    for item in synthetic_data
]

synthetic_dataset = Dataset.from_list(formatted_data)

def compute_perplexity(example):
    inputs = tokenizer(
        images=None,
        text=example["formatted_text"],
        truncation=True,
        max_length=2048,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss.item()
        perplexity = math.exp(loss) if loss < 100 else float("inf")
    return {"perplexity": perplexity}

synthetic_dataset = synthetic_dataset.map(compute_perplexity)
perplexities = synthetic_dataset["perplexity"]
avg_perplexity = sum(perplexities) / len(perplexities)

print(f"{avg_perplexity:.2f}")
