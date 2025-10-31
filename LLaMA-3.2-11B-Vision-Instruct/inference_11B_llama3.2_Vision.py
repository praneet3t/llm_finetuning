import torch
import time
from unsloth import FastLanguageModel
from transformers import TextStreamer

# === Load your fine-tuned model (text-only inference from a vision model) ===
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./llama3_oceanbench_lora",  # Path to LoRA-fine-tuned model
    max_seq_length=4096,
    load_in_4bit=True,
    device_map="auto"
)

# === Ensure CUDA is available ===
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")
print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# === Enable inference mode ===
FastLanguageModel.for_inference(model)

# === Setup streamer (optional for streaming responses) ===
use_streamer = True
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True) if use_streamer else None

print("\nType your audit question below. Type 'exit' to quit.\n")

# === Inference loop ===
while True:
    question = input("You: ").strip()
    if question.lower() == "exit":
        print("Exiting...")
        break

    # === Chat template formatting ===
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # === Tokenize with explicit text-only input ===
    inputs = tokenizer(
        images=None,                 # Required: no image input
        text=prompt,                 # Explicitly named to avoid confusion
        return_tensors="pt"
    ).to(model.device)

    # === Generate response ===
    start = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
        streamer=streamer,
        pad_token_id=tokenizer.pad_token_id
    )
    end = time.time()

    # === Decode and print only if streamer is disabled ===
    if not use_streamer:
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nResponse:\n{response.strip()}\n")

    print(f"Inference time: {end - start:.2f} seconds\n")

# === Example prompts for testing ===
# 1. Why is Sea Surface Height (SSH) harder to reconstruct than SST in some OceanBench tasks?
# 2. Explain the role of Argo floats in oceanography.
# 3. What are the advantages of dynamic ocean routing systems?
# 4. Describe the impact of ocean current variability on shipping fuel consumption.
# 5. How do satellite altimeters help in real-time ocean monitoring?
