import torch
import time
from unsloth import FastLanguageModel

# === Load the fine-tuned model ===
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./finetuned_model",         # path to your saved model
    max_seq_length=2048,
    load_in_4bit=True,
    device_map="auto"
)

# === Check if CUDA is available ===
assert torch.cuda.is_available(), "CUDA is not available!"
device_name = torch.cuda.get_device_name(0)
print(f" GPU: {device_name}")

# === Interactive Inference Loop ===
print("\n📌 Type your question below. Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    if user_input.strip().lower() == "exit":
        print("Exiting....")
        break

    # Apply chat formatting
    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_input}], tokenize=False
    )

    # Tokenize and move to GPU
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")

    # Measure inference time
    start = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )
    end = time.time()

    # Decode and print response
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print(f"\nLlama4:\n{response.strip()}")
    print(f"\n⏱️ Inference Time: {end - start:.2f} seconds\n")

# === Example prompts for testing ===
# 1. Why is Sea Surface Height (SSH) harder to reconstruct than SST in some OceanBench tasks?
# 2. Explain the role of Argo floats in oceanography.
# 3. What are the advantages of dynamic ocean routing systems?
# 4. Describe the impact of ocean current variability on shipping fuel consumption.
# 5. How do satellite altimeters help in real-time ocean monitoring?
