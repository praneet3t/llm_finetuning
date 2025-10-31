import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
from peft import PeftModel
from bert_score import score
from tqdm import tqdm

logging.set_verbosity_error()

# === Configuration ===
models = [
    {
        "name": "Qwen3-14B",
        "base_id": "unsloth/Qwen3-14B-unsloth-bnb-4bit",
        "adapter_path": "/home/incois/tvsubhaskar/llm_project/Qwen3-14B/qwen3_14b_oceanbench_lora",
    },
    # {
    #     "name": "Qwen3-32B",
    #     "base_id": "unsloth/Qwen3-32B-unsloth-bnb-4bit",
    #     "adapter_path": "/home/incois/tvsubhaskar/llm_project/Qwen3-32B/qwen3_32b_oceanbench_lora",
    # },
    # ... other models left commented out ...
]

DATA_PATH      = "/home/incois/tvsubhaskar/llm_project/synthetic.json"  # JSON array of {"question","answer"}
MAX_NEW_TOKENS = 128
NUM_EXAMPLES   = 5     # only first 5 for debugging
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

# --- load synthetic data as JSON array ---
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

questions  = [ex["question"] for ex in data[:NUM_EXAMPLES]]
references = [ex["answer"]   for ex in data[:NUM_EXAMPLES]]

# --- helper to format prompt ---
def make_prompt(q):
    return f"User: {q}\nAssistant: Answer succinctly without any internal reasoning or commentary:"

# === loop over just the first model for now ===
for cfg in models[:1]:
    print(f"\n### Evaluating BERTScore for {cfg['name']} on {NUM_EXAMPLES} samples ###")

    # 1) Load and merge base + LoRA
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_id"],
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map="auto"
    ).to(DEVICE)
    tuned = PeftModel.from_pretrained(base, cfg["adapter_path"]).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_id"])
    tuned.eval()

    # 2) Generate predictions
    preds = []
    for q in tqdm(questions, desc=f"{cfg['name']} gen"):
        prompt = make_prompt(q)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(DEVICE)

        with torch.no_grad():
            out = tuned.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        # slice off the prompt tokens
        text = tokenizer.decode(
            out[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        ).strip()
        preds.append(text)

    # 3) Compute BERTScore
    P, R, F1 = score(
        cands=preds,
        refs=references,
        lang="en",
        model_type="microsoft/deberta-v3-small",  # likely cached offline
        rescale_with_baseline=True,
    )

    print(f"\n{cfg['name']} BERTScore (F1): {F1.mean().item():.4f}")
    print(f" Precision: {P.mean().item():.4f},  Recall: {R.mean().item():.4f}")
