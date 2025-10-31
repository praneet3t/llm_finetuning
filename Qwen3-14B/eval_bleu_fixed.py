import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
from sacrebleu import corpus_bleu

logging.set_verbosity_error()

# Paths / IDs
base_model_id = "unsloth/Qwen3-14B-unsloth-bnb-4bit"                   # your original Qwen3‑14B repo
adapter_path  = "./qwen3_14b_oceanbench_lora"       # your LoRA folder
data_path     = "/home/incois/tvsubhaskar/llm_project/synthetic_data.json"

# 1) Load base Qwen3‑14B (4‑bit) and its tokenizer
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# 2) Apply LoRA adapters on top
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

# 3) Load dataset
dataset = load_dataset("json", data_files=data_path, split="train")

# 4) Select 5 examples for quick testing
test_ds = dataset
#.select(range(5))

# 5) Build prompts & references
prompts    = [f"User: {q}\nAssistant:" for q in test_ds["input"]]
references = [ test_ds["output"] ]  # single list of 5 reference strings

# 6) Generate & collect predictions
predictions = []
for prompt in tqdm(prompts, desc="Generating"):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    gen_tokens = gen_ids[0][ inputs["input_ids"].shape[-1] : ]
    predictions.append(tokenizer.decode(gen_tokens, skip_special_tokens=True).strip())

# 7) Compute BLEU
bleu = corpus_bleu(predictions, references)
print(f"BLEU Score: {bleu.score:.2f}")

# --- To run on all 240 examples, replace the test_ds lines with: ---
# prompts    = [f"User: {q}\nAssistant:" for q in dataset["input"]]
# references = [ dataset["output"] ]
