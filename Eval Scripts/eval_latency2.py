import warnings
import torch
import time
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────
# 1) Silence unwanted warnings
warnings.filterwarnings("ignore", category=UserWarning, module="peft.peft_model")
warnings.filterwarnings("ignore", category=UserWarning, module="bitsandbytes.nn.modules")
logging.set_verbosity_error()
# ──────────────────────────────────────────────────────────────────────────

# === Configuration ===
models = [
    {
        "name": "Llama 3.2 11B Vision Instruct",
        "base_id": "unsloth/Llama-3.2-11B-Vision-Instruct",
        "adapter_path": "/home/incois/tvsubhaskar/llm_project/LLaMA-3.2-11B-Vision-Instruct/llama3_oceanbench_lora",
    },
    {
        "name": "Llama 3.2 3B Instruct",
        "base_id": "unsloth/Llama-3.2-3B-Instruct",
        "adapter_path": "/home/incois/tvsubhaskar/llm_project/LLaMA-3.2-3B-Instruct/finetuned_model",
    },
]

data_path   = "/home/incois/tvsubhaskar/llm_project/synthetic.json"
NUM_PROMPTS = 100     # set to 100–200 for publication; 5 for a quick test
MAX_TOKENS  = 64    # keeps generated answers reasonably short

# === Load & Sample Prompts ===
ds = load_dataset("json", data_files=data_path, split="train")
if NUM_PROMPTS < len(ds):
    ds = ds.shuffle(seed=42).select(range(NUM_PROMPTS))
prompts = [f"User: {q}\nAssistant:" for q in ds["question"]]

# === Benchmark Function ===
def benchmark_latency(model_name, base_id, adapter_path, prompts):
    print(f"\n→ {model_name}")

    # 1) Load base model in 4‑bit + apply LoRA adapters
    base = AutoModelForCausalLM.from_pretrained(
        base_id,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(base_id)

    # 2) Warm‑up (cold loading overhead)
    warm_inp = tokenizer(prompts[0], return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        _ = model.generate(**warm_inp, max_new_tokens=10)

    # 3) Measure per‑prompt latency
    latencies = []
    for prompt in tqdm(prompts, desc=f"  Generating ({model_name})"):
        inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            _ = model.generate(**inp, max_new_tokens=MAX_TOKENS, do_sample=False)
        torch.cuda.synchronize()
        t1 = time.time()
        latencies.append((t1 - t0) * 1000)

    arr = np.array(latencies)
    mean, ci95 = arr.mean(), 1.96 * arr.std() / np.sqrt(len(arr))
    print(f"  Avg Latency: {mean:.1f} ms  ± {ci95:.1f} ms (95% CI)")

# === Run Benchmarks ===
for m in models:
    benchmark_latency(m["name"], m["base_id"], m["adapter_path"], prompts)
