#!/usr/bin/env python
import json
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorForLanguageModeling

def main():
    # 1. Load and quantize the base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-14B-unsloth-bnb-4bit",
        #unsloth/Qwen3-32B-unsloth-bnb-4bit
        max_seq_length=4096,
        load_in_4bit=True,
        full_finetuning=False,
        device_map="auto",
    )

    # 2. Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=[
            "q_proj","k_proj","v_proj","o_proj",
            "gate_proj","up_proj","down_proj"
        ],
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # 3. Load and format OceanBench Q&A
    ds = load_dataset("zjunlp/OceanBench", split="train")

    def format_batch(examples):
        out = {"text": []}
        for entry in examples["text"]:
            obj = json.loads(entry)
            prompt = tokenizer.apply_chat_template([
                {"role": "user", "content": obj.get("input", "")},
                {"role": "assistant", "content": obj.get("output", "")}
            ], tokenize=False)
            out["text"].append(prompt)
        return out

    ds = ds.map(format_batch, batched=True, remove_columns=ds.column_names)

    # 4. Enable training mode
    FastLanguageModel.for_training(model)

    # 5. Set up trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=8,    # A100 80 GB handles batch of 8
            gradient_accumulation_steps=2,    # effective batch = 16
            warmup_steps=100,
            num_train_epochs=2,
            learning_rate=2e-5,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
            bf16=torch.cuda.is_bf16_supported(),  # use bf16 on A100
            seed=42,
            report_to="none",
        ),
    )

    # 6. Train
    trainer.train()

    # 7. Save LoRA adapters
    model.save_pretrained("qwen3_14b_oceanbench_lora")
    tokenizer.save_pretrained("qwen3_14b_oceanbench_lora")


if __name__ == "__main__":
    main()
