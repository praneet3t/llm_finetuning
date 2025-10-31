import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
from sacrebleu import corpus_bleu

logging.set_verbosity_error()

base_model_id = "unsloth/Phi-4-unsloth-bnb-4bit"
adapter_path  = "./phi4_oceanbench_lora"
data_path     = "/home/incois/tvsubhaskar/llm_project/synthetic_data.json"

# Load base Phi‑4 and apply LoRA adapters
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# Load full synthetic QA dataset
dataset = load_dataset("json", data_files=data_path, split="train")
prompts = [
    f"System: You are a concise assistant. Answer in one sentence.\nUser: {q}\nAssistant:"
    for q in dataset["input"]
]
references = [ dataset["output"] ]  # single-list format

# Generate predictions over all 240 examples
predictions = []
for prompt in tqdm(prompts, desc="Generating responses"):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(model.device)
    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    gen_tokens = gen_ids[0][ inputs["input_ids"].shape[-1] : ]
    predictions.append(tokenizer.decode(gen_tokens, skip_special_tokens=True).strip())

# Compute and print BLEU
bleu = corpus_bleu(predictions, references)
print(f"BLEU Score (240 examples): {bleu.score:.2f}")
