import torch
import math
from datasets import load_dataset
from unsloth import FastLanguageModel

# — Load LoRA-fine-tuned model (vision checkpoint, but we force text-only) —
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./llama3_oceanbench_lora",
    max_seq_length=2048,
    load_in_4bit=True,
    device_map="auto"
)
model.eval()

# — Pull in the dataset and format it —
dataset = load_dataset("zjunlp/OceanBench", split="train")

import json
def extract_text(example):
    try:
        parsed = json.loads(example["text"])
        return {"formatted_text": f"User: {parsed['input']}\nAssistant: {parsed['output']}"}
    except:
        return {"formatted_text": "Invalid format"}

dataset = dataset.map(extract_text, remove_columns=dataset.column_names)

# — Corrected perplexity function for a vision-language tokenizer used in text-only mode —
def compute_perplexity(example):
    # force the processor to only consume text
    inputs = tokenizer(
        images=None,                       # explicitly no image
        text=example["formatted_text"],    # named argument to disambiguate
        truncation=True,
        max_length=2048,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss.item()
        # guard against overflow
        ppl = math.exp(loss) if loss < 100 else float("inf")
    return {"perplexity": ppl}

# — Map, gather, and average —
dataset = dataset.map(compute_perplexity)
perplexities = dataset["perplexity"]
avg_perplexity = sum(perplexities) / len(perplexities)
print(f"\nAverage Perplexity (training set): {avg_perplexity:.2f}")

#Average Perplexity (training set): 19.16