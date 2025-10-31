import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
from datasets import load_dataset
from tqdm import tqdm

logging.set_verbosity_error()

# === Config ===
BASE_MODEL_ID = "unsloth/Qwen3-32B-unsloth-bnb-4bit"
DATA_FILE     = "/home/incois/tvsubhaskar/llm_project/factual_accuracy_Llama-3.2-11B-Vision-Instruct.jsonl"
MAX_NEW_TOKENS = 2   # just enough to generate "Correct" or "Incorrect"

# === Load entries ===
entries = [json.loads(line) for line in open(DATA_FILE)]

# === Load base Qwen3‑14B ===
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
model.eval()

# === Judge loop ===
results = []
for ex in tqdm(entries, desc="Judging factual correctness"):
    prompt = (
        "You are an expert fact‑checker.\n"
        "Read the Question, Model Answer, and Reference Answer.\n"
        "Reply with exactly Correct or Incorrect.\n\n"
        f"Question: {ex['question']}\n"
        f"Model Answer: {ex['model_answer']}\n"
        f"Reference Answer: {ex['reference_answer']}\n\n"
        "Verdict:"
    )

    tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    with torch.no_grad():
        gen = model.generate(
            **tokens,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    verdict = tokenizer.decode(
        gen[0][ tokens["input_ids"].shape[-1] : ],
        skip_special_tokens=True
    ).strip()

    results.append({**ex, "label": verdict})

# === Compute overall accuracy ===
correct = sum(1 for r in results if r["label"].lower()=="correct")
total   = len(results)
print(f"\nFactual Accuracy (base Qwen3‑14B): {correct}/{total} = {correct/total:.2%}")

# === (Optional) save labeled results ===
with open("qwen_base_factual_labels.jsonl", "w") as fout:
    for r in results:
        fout.write(json.dumps(r) + "\n")
