import torch
import time
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm

logging.set_verbosity_error()

# === Configuration ===
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
]

data_path    = "/home/incois/tvsubhaskar/llm_project/synthetic.json"
NUM_PROMPTS  = 100    # for publication, use 100–200; for quick test, use 5
MAX_TOKENS   = 64     # cap generation to keep answers short

# === Load & Sample Prompts ===
ds = load_dataset("json", data_files=data_path, split="train")
if NUM_PROMPTS < len(ds):
    ds = ds.shuffle(seed=42).select(range(NUM_PROMPTS))
prompts = [f"User: {q}\nAssistant:" for q in ds["question"]]

def benchmark_latency(model_name, base_id, adapter_path, prompts):
    print(f"\n→ {model_name}")
    # 1) Load base + LoRA adapter
    base = AutoModelForCausalLM.from_pretrained(
        base_id,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(base_id)

    # 2) Activation step: send a simple "Hi" to wake the model
    activation_prompt = "User: Hi\nAssistant:"
    act_inp = tokenizer(activation_prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        _ = model.generate(**act_inp, max_new_tokens=5)

    # 3) Now measure latency on the real prompts
    latencies = []
    for prompt in tqdm(prompts, desc=f"  Generating ({model_name})"):
        inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            _ = model.generate(**inp, max_new_tokens=MAX_TOKENS, do_sample=False)
        torch.cuda.synchronize()
        t1 = time.time()
        latencies.append((t1 - t0) * 1000)  # ms

    arr = np.array(latencies)
    mean, ci95 = arr.mean(), 1.96 * arr.std() / np.sqrt(len(arr))
    print(f"  Avg Latency: {mean:.1f} ms ± {ci95:.1f} ms (95% CI)")

# === Run Benchmarks ===
for m in models:
    benchmark_latency(m["name"], m["base_id"], m["adapter_path"], prompts)
