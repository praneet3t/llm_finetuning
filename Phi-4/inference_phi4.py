#!/usr/bin/env python
# inference_phi4_oceanbench.py

import time
import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer

def main():
    # Load the fine-tuned LoRA-adapted Phi‑4 model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = "phi4_oceanbench_lora",
        max_seq_length = 2048,
        load_in_4bit   = True,
        device_map     = "auto",
    )

    # Switch to inference mode
    FastLanguageModel.for_inference(model)

    # Simple oceanbench-style chat template
    #    (must match your training template)
    def make_prompt(question: str, context: str) -> str:
        return (
            "Below is an oceanographic question and its context. "
            "Provide a concise, accurate answer.\n\n"
            f"### Question:\n{question}\n\n"
            f"### Context:\n{context}\n\n"
            "### Answer:\n"
        )

    print("\nEnter a question (or type 'exit' to quit):\n")
    while True:
        q = input("Question: ").strip()
        if q.lower() == "exit":
            break
        ctx = input("Context (or blank to reuse question): ").strip() or q

        prompt = make_prompt(q, ctx)
        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        start = time.time()
        _ = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.8,
            streamer=streamer,
            pad_token_id=tokenizer.pad_token_id,
        )
        elapsed = time.time() - start

        print(f"\nInference time: {elapsed:.2f}s\n")

if __name__ == "__main__":
    main()
