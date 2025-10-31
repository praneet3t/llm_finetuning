import os
os.environ["TRITON_DISABLE_LINE_INFO"] = "1"

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth.chat_templates import get_chat_template, standardize_sharegpt

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)

tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

from datasets import load_dataset

import json

# === Load dataset ===
dataset = load_dataset("zjunlp/OceanBench", split="train")

# === Format dataset like ShareGPT style using chat template ===
def format_batch(examples):
    return {
        "text": [
            tokenizer.apply_chat_template([
                {"role": "user", "content": json.loads(entry)["input"]},
                {"role": "assistant", "content": json.loads(entry)["output"]}
            ], tokenize=False)
            for entry in examples["text"]
        ]
    }

dataset = dataset.map(format_batch, batched=True)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=TrainingArguments(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        warmup_steps=100,
        max_steps=2000,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="paged_adamw_8bit",
        output_dir="outputs"
    ),
)

trainer.train()

model.save_pretrained("finetuned_model")


