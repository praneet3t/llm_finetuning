import json
import torch
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
    {
        "name": "Llama-3.2-11B-Vision-Instruct",
        "base_id": "unsloth/Llama-3.2-11B-Vision-Instruct",
        "adapter_path": "/home/incois/tvsubhaskar/llm_project/LLaMA-3.2-11B-Vision-Instruct/llama3_oceanbench_lora",
    },
    {
        "name": "Llama-3.2-3B-Instruct",
        "base_id": "unsloth/Llama-3.2-3B-Instruct",
        "adapter_path": "/home/incois/tvsubhaskar/llm_project/LLaMA-3.2-3B-Instruct/finetuned_model",
    },
]

data_path    = "/home/incois/tvsubhaskar/llm_project/synthetic_data.json"
NUM_EXAMPLES = 100    # 50–200 recommended
MAX_TOKENS   = 64

# === Load & sample once ===
ds = load_dataset("json", data_files={"data": data_path}, split="data")
if NUM_EXAMPLES < len(ds):
    ds = ds.shuffle(seed=42).select(range(NUM_EXAMPLES))

def extract_fields(ex):
    return {
        "input": ex.get("question", ex.get("input")),
        "output": ex.get("answer", ex.get("output"))
    }
ds = ds.map(extract_fields, remove_columns=ds.column_names)

# === Loop and generate one .jsonl per model ===
for m in models:
    out_file = f"factual_accuracy_{m['name'].replace(' ', '_')}.jsonl"
    print(f"\n→ Generating factual‐accuracy prompts for {m['name']} → {out_file}")

    # load model + adapter + tokenizer
    base = AutoModelForCausalLM.from_pretrained(
        m["base_id"], load_in_4bit=True, torch_dtype=torch.float16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, m["adapter_path"])
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(m["base_id"])

    with open(out_file, "w") as fout:
        for ex in tqdm(ds, desc=m["name"]):
            question = ex["input"]
            reference = ex["output"]

            # generate the model’s answer
            prompt = f"User: {question}\nAssistant:"
            toks   = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
            with torch.no_grad():
                gen_ids = model.generate(**toks, max_new_tokens=MAX_TOKENS, do_sample=False)
            gen_text = tokenizer.decode(gen_ids[0][toks["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

            # build the evaluation prompt
            eval_prompt = (
                "Given the question and answer, determine if the answer is factually correct. "
                "Respond with \"Correct\" or \"Incorrect\".\n\n"
                f"Question: {question}\n"
                f"Model Answer: {gen_text}\n"
                f"Reference/Expected Answer: {reference}"
            )

            # write one JSON object per line (JSONL)
            fout.write(json.dumps({
                "model": m["name"],
                "question": question,
                "model_answer": gen_text,
                "reference_answer": reference,
                "eval_prompt": eval_prompt
            }) + "\n")

    print(f"  ✔️  Wrote {NUM_EXAMPLES} entries to {out_file}")
