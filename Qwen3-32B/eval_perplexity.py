import torch
from datasets import load_dataset
import math
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./qwen3_32b_oceanbench_lora",  
    max_seq_length=2048,
    load_in_4bit=True,
    device_map="auto"
)
model.eval()
dataset = load_dataset("zjunlp/OceanBench", split="train")  

import json
def extract_text(example):
    try:
        parsed = json.loads(example["text"])
        return {"formatted_text": f"User: {parsed['input']}\nAssistant: {parsed['output']}"}
    except:
        return {"formatted_text": "Invalid format"}

dataset = dataset.map(extract_text)

# Perplexity computation 
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
dataset = dataset.map(compute_perplexity)
perplexities = dataset["perplexity"]
avg_perplexity = sum(perplexities) / len(perplexities)

print(f"\n Average Perplexity (using training data): {avg_perplexity:.2f}")
# Average Perplexity (using training data): 5.89