import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
from bert_score import score

logging.set_verbosity_error()

# === Configuration ===
MAX_NEW_TOKENS = 64
models = [
    {
        "name": "Qwen3-14B",
        "base_id": "unsloth/Qwen3-14B-unsloth-bnb-4bit",
        "adapter_path": "/home/incois/tvsubhaskar/llm_project/Qwen3-14B/qwen3_14b_oceanbench_lora"
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
        "adapter_path": "/home/incois/tvsubhaskar/llm_project/LLaMA-3.2-11B-Vision-Instruct/llama3_oceanbench_lora"
    },
    {
        "name": "Llama-3.2-3B-Instruct",
        "base_id": "unsloth/Llama-3.2-3B-Instruct",
        "adapter_path": "/home/incois/tvsubhaskar/llm_project/LLaMA-3.2-3B-Instruct/finetuned_model",
    },
]

data_path = "/home/incois/tvsubhaskar/llm_project/synthetic_data.json"

# === 1) Load & parse synthetic dataset ===
ds = load_dataset("json", data_files={"data": data_path}, split="data")

def extract_fields(ex):
    return {
        "input": ex.get("question", ex.get("input")),
        "output": ex.get("answer", ex.get("output"))
    }

ds = ds.map(extract_fields, remove_columns=ds.column_names)

# === Loop through models ===
for m in models:
    print(f"\n=== Evaluating {m['name']} ===")

    # === Load model + tokenizer ===
    base_model = AutoModelForCausalLM.from_pretrained(
        m['base_id'], load_in_4bit=True, torch_dtype=torch.float16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, m['adapter_path'])
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(m['base_id'])

    # === Generate & collect across entire synthetic dataset ===
    predictions = []
    references  = []
    for ex in tqdm(ds, desc=f"Generating ({m['name']})"):
        prompt = f"User: {ex['input']}\nAssistant:"
        toks   = tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=2048
        ).to(model.device)

        with torch.no_grad():
            gen = model.generate(
                **toks,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False
            )

        # Extract only newly generated tokens
        new_tokens = gen[0][toks["input_ids"].shape[-1]:]
        out = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        predictions.append(out)
        references.append(ex['output'])

    # === Compute BERTScore ===
    P, R, F1 = score(
        cands=predictions,
        refs=references,
        lang="en",
        model_type="microsoft/deberta-xlarge-mnli",
        verbose=False,
        rescale_with_baseline=True,
    )

    print(f"\nBERTScore for {m['name']} — Precision: {P.mean():.3f}, Recall: {R.mean():.3f}, F1: {F1.mean():.3f}")
