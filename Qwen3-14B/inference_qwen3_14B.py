#!/usr/bin/env python
import time
import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer

def main():
    # 1. Load LoRA-adapted model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="./qwen3_14b_oceanbench_lora",
        max_seq_length=4096,
        load_in_4bit=True,
        device_map="auto",
    )

    # 2. Switch to inference mode
    FastLanguageModel.for_inference(model)

    # 3. Check GPU
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # 4. Prepare streamer (optional)
    use_streamer = True
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True) if use_streamer else None

    print("\nType your question. Type 'exit' to quit.\n")
    while True:
        query = input("You: ").strip()
        if query.lower() == "exit":
            break

        # 5. Format prompt
        messages = [{"role": "user", "content": query}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 6. Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # 7. Generate
        start = time.time()
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            streamer=streamer,
            pad_token_id=tokenizer.pad_token_id,
        )
        end = time.time()

        # 8. Print (if not streaming)
        if not use_streamer:
            resp = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"\nResponse:\n{resp.strip()}\n")

        print(f"Inference time: {end - start:.2f} seconds\n")

if __name__ == "__main__":
    main()

# === Example prompts for testing ===
# 1. Why is Sea Surface Height (SSH) harder to reconstruct than SST in some OceanBench tasks?
# 2. Explain the role of Argo floats in oceanography.
# 3. What are the advantages of dynamic ocean routing systems?
# 4. Describe the impact of ocean current variability on shipping fuel consumption.
# 5. How do satellite altimeters help in real-time ocean monitoring?
