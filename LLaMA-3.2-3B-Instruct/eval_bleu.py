import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
from sacrebleu import corpus_bleu

logging.set_verbosity_error()

base_model_id = "unsloth/Llama-3.2-3B-Instruct"
adapter_path  = "./finetuned_model"
data_path     = "/home/incois/tvsubhaskar/llm_project/synthetic_data.json"

# load base + LoRA adapters
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(base_model_id)

dataset = load_dataset("json", data_files=data_path, split="train")

# select 5 examples for a quick test
test_dataset = dataset.select(range(10))

prompts    = [f"User: {q}\nAssistant:" for q in test_dataset["input"]]
references = [ test_dataset["output"] ]  # sacreBLEU expects [list_of_reference_strings]

predictions = []
for prompt in tqdm(prompts, desc="Generating responses"):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    gen_tokens = gen_ids[0][ inputs["input_ids"].shape[-1] : ]
    predictions.append(tokenizer.decode(gen_tokens, skip_special_tokens=True).strip())

bleu = corpus_bleu(predictions, references)
print(f"BLEU Score : {bleu.score:.2f}")

# To run on the full dataset instead of 5, replace:
#   test_dataset = dataset.select(range(5))
# with:
#   test_dataset = dataset
