#!/usr/bin/env python
import json
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorForLanguageModeling

def main():
    # 1️⃣ Load & quantize Phi‑4 base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Phi-4-unsloth-bnb-4bit",
        max_seq_length=2048,
        load_in_4bit=True,
        device_map="auto",
    )

    # 2️⃣ Attach LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj","k_proj","v_proj","o_proj",
            "gate_proj","up_proj","down_proj"
        ],
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # 3️⃣ Load & format OceanBench Q&A
    ds = load_dataset("zjunlp/OceanBench", split="train")
    def format_batch(examples):
        out = {"text": []}
        for entry in examples["text"]:
            obj = json.loads(entry)
            prompt = tokenizer.apply_chat_template([
                {"role": "user",      "content": obj.get("input", "")},
                {"role": "assistant", "content": obj.get("output", "")}
            ], tokenize=False)
            out["text"].append(prompt)
        return out

    ds = ds.map(format_batch, batched=True, remove_columns=ds.column_names)

    # 4️⃣ Switch to training mode
    FastLanguageModel.for_training(model)

    # 5️⃣ Configure trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=4,    # A100 handles 4–8 easily at 2048 tokens
            gradient_accumulation_steps=4,    # effective batch size = 16
            warmup_steps=100,
            num_train_epochs=3,
            learning_rate=1e-4,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
            bf16=torch.cuda.is_bf16_supported(),  # use bf16 on A100
            seed=42,
            output_dir="phi4_oceanbench_lora",
            report_to="none",
        ),
    )

    # 6️⃣ Train
    print("▶️ Starting fine‑tuning Phi‑4 on OceanBench…")
    trainer.train()

    # 7️⃣ Save LoRA adapters + tokenizer
    model.save_pretrained("phi4_oceanbench_lora")
    tokenizer.save_pretrained("phi4_oceanbench_lora")
    print("✅ Adapters saved to ./phi4_oceanbench_lora")

if __name__ == "__main__":
    main()
