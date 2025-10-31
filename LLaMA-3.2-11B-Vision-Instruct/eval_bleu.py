import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
from sacrebleu import corpus_bleu

logging.set_verbosity_error()

# 1) Base LLaMA‑3.2‑3B‑Instruct repo
base_model_id = "unsloth/Llama-3.2-11B-Vision-Instruct"

# 2) Load base model in 4‑bit
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 3) Apply your LoRA adapters
model = PeftModel.from_pretrained(base_model, "./llama3_oceanbench_lora")
model.eval()

# 4) Matching tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# 5) Load evaluation dataset
dataset = load_dataset(
    "json",
    data_files="/home/incois/tvsubhaskar/llm_project/synthetic_data.json",
    split="train"
)
prompts    = dataset["input"]
references = [[ref] for ref in dataset["output"]]

# 6) Generate and collect predictions
predictions = []
for prompt in tqdm(prompts, desc="Generating responses"):
    inputs = tokenizer(
        images=None,              # vision‑instruct models require this arg
        text=prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(model.device)

    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )

    gen_tokens = gen_ids[0][ inputs["input_ids"].shape[-1] : ]
    pred = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    predictions.append(pred)

# 7) Compute BLEU
bleu = corpus_bleu(predictions, references)
print(f"BLEU Score: {bleu.score:.2f}")
