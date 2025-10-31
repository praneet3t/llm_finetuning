import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
from datasets import load_dataset
from tqdm import tqdm
from sacrebleu import corpus_bleu

logging.set_verbosity_error()

model_path = "./qwen3_32b_oceanbench_lora"
dataset_path = "/home/incois/tvsubhaskar/llm_project/synthetic_data.json"

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

dataset = load_dataset("json", data_files=dataset_path, split="train")

prompts = dataset["input"]
references = [[ref] for ref in dataset["output"]]

predictions = []
for prompt in tqdm(prompts, desc="Generating responses"):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    pred = tokenizer.decode(gen_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    predictions.append(pred)

bleu = corpus_bleu(predictions, references)
print(f"BLEU Score: {bleu.score:.2f}")
