import torch, json, time
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorForLanguageModeling

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-11B-Vision-Instruct",
    load_in_4bit = True,
    use_gradient_checkpointing = "unsloth",
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 64,
    lora_alpha = 64,
    lora_dropout = 0.05,
    bias = "none",
    use_rslora = False,
    random_state = 42,
    target_modules = "all-linear",
)

dataset = load_dataset("zjunlp/OceanBench", split="train")

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
dataset = dataset.remove_columns([col for col in dataset.column_names if col != "text"])

FastLanguageModel.for_training(model)

gpu_stats = torch.cuda.get_device_properties(0)
max_gpu_memory = round(gpu_stats.total_memory / 1024**3, 3)  # GB
start_reserved = round(torch.cuda.max_memory_reserved() / 1024**3, 3)

start_time = time.time()
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False),
    args = SFTConfig(
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 2,
        warmup_steps = 100,
        logging_steps = 10,
        num_train_epochs = 2,
        learning_rate = 2e-5,
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        optim = "adamw_8bit",
        max_seq_length = 4096,
        output_dir = "llama3_oceanbench_lora",
        save_strategy = "epoch",
        save_total_limit = 2,
        bf16 = torch.cuda.is_bf16_supported(),
        report_to = "none",
        seed = 42,
    ),
)

# === Training ===
train_output = trainer.train()

# === Timer end ===
end_time = time.time()

# === Measure memory after training ===
end_reserved = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
reserved_diff = round(end_reserved - start_reserved, 3)
reserved_pct = round(end_reserved / max_gpu_memory * 100, 3)
reserved_diff_pct = round(reserved_diff / max_gpu_memory * 100, 3)

# === Print training stats ===
print(f"\nTraining completed.")
print(f" {round(end_time - start_time, 3)} seconds used for training.")
print(f" {round((end_time - start_time) / 60, 2)} minutes used for training.")
print(f" Peak reserved memory = {end_reserved} GB.")
print(f" Peak reserved memory for training = {reserved_diff} GB.")
print(f" Peak reserved memory % of max memory = {reserved_pct} %.")
print(f" Peak reserved memory for training % of max memory = {reserved_diff_pct} %.")

# === Save model ===
model.save_pretrained("llama3_oceanbench_lora")
tokenizer.save_pretrained("llama3_oceanbench_lora")
