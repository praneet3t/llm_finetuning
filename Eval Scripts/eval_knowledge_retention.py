import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
from peft import PeftModel
from tqdm import tqdm

logging.set_verbosity_error()

# === Models Configuration ===
models = [
    {
        "name": "Qwen3-14B",
        "base_id": "unsloth/Qwen3-14B-unsloth-bnb-4bit",
        "adapter_path": "/home/incois/tvsubhaskar/llm_project/Qwen3-14B/qwen3_14b_oceanbench_lora",
    },
    {
        "name": "Qwen3-32B",
        "base_id": "unsloth/Qwen3-32B-unsloth-bnb-4bit",
        "adapter_path": "/home/incois/tvsubhaskar/llm_project/Qwen3-32B/qwen3_32b_oceanbench_lora",
    },
    {
        "name": "Phi-4",
        "base_id": "unsloth/Phi-4-unsloth-bnb-4bit",
        "adapter_path": "/home/incois/tvsubhaskar/llm_project/Phi-4/phi4_oceanbench_lora",
    },
    {
        "name": "Llama-3.2-11B-Vision-Instruct",
        "base_id": "unsloth/Llama-3.2-11B-Vision-Instruct",
        "adapter_path": "/home/incois/tvsubhaskar/llm_project/LLaMA-3.2-11B-Vision-Instruct/llama3_oceanbench_lora",
    },
    {
        "name": "Llama-3.2-3B-Instruct",
        "base_id": "unsloth/Llama-3.2-3B-Instruct",
        "adapter_path": "/home/incois/tvsubhaskar/llm_project/LLaMA-3.2-3B-Instruct/finetuned_model",
    },
]

data_path      = "/home/incois/tvsubhaskar/llm_project/knowledge_retention_trivia.jsonl"
MAX_NEW_TOKENS = 64

# —————————————————————————————————————————————————————————————————————
# Normalization for matching
def normalize(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return text

# Load the full trivia set
trivia = []
with open(data_path, 'r', encoding='utf-8') as f:
    for line in f:
        obj = json.loads(line)
        trivia.append({"question": obj["question"], "answer": obj["answer"]})
total = len(trivia)

# Evaluation routine using substring match
def evaluate_hit_rate(model, tokenizer, device, tag):
    hits = 0
    for qa in tqdm(trivia, desc=f"{tag} eval"):
        prompt = (
            f"User: {qa['question']}\n"
            "Assistant: Answer succinctly without any internal reasoning or commentary:"
        )
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        pred = tokenizer.decode(
            gen_ids[0][inputs['input_ids'].shape[-1]:],
            skip_special_tokens=True
        ).strip()

        # keyword‐based match
        if normalize(qa['answer']) in normalize(pred):
            hits += 1

    return hits / total * 100

# —————————————————————————————————————————————————————————————————————
summary = []

for cfg in models:
    print(f"\n=== Evaluating {cfg['name']} ===")

    # 1) Load base model
    base = AutoModelForCausalLM.from_pretrained(
        cfg['base_id'], load_in_4bit=True, torch_dtype=torch.float16, device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg['base_id'])
    device   = next(base.parameters()).device
    base.eval()
    hit_base = evaluate_hit_rate(base, tokenizer, device, f"{cfg['name']} [base]")

    # 2) Load LoRA‑tuned model
    tuned = PeftModel.from_pretrained(base, cfg['adapter_path'])
    tuned_device = next(tuned.parameters()).device
    tuned.eval()
    hit_tuned = evaluate_hit_rate(tuned, tokenizer, tuned_device, f"{cfg['name']} [tuned]")

    retention = (hit_tuned / hit_base * 100) if hit_base > 0 else 0.0
    print(f"\n{cfg['name']} → Base Hit‑Rate: {hit_base:.2f}% | Tuned: {hit_tuned:.2f}% | Retention: {retention:.2f}%")

    summary.append({
        "model": cfg['name'],
        "HitRate_base_%": hit_base,
        "HitRate_tuned_%": hit_tuned,
        "Retention_%": retention
    })

# Save summary
with open('knowledge_retention_summary.json', 'w') as fout:
    json.dump(summary, fout, indent=2)
#Llama-3.2-3B-Instruct → Base Hit‑Rate: 65.50% | Tuned: 74.50% | Retention: 113.74%