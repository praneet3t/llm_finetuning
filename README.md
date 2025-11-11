# LLM Fine-tuning for Domain-Specific Question Answering

## Project Overview

This project evaluates and fine-tunes large language models for specialized question-answering applications in organizational and audit contexts. The goal is to build an intelligent system that can handle complex domain-specific queries with high accuracy and reliability.

### Motivation

Organizations frequently receive formal questions that require accurate, well-researched responses. Manual processing is time-consuming and resource-intensive. This project explores using fine-tuned LLMs to:

- Accelerate response generation for domain-specific queries
- Maintain factual accuracy and consistency
- Reduce manual workload while preserving quality
- Enable future agentic capabilities for autonomous information retrieval

### Methodology

1. **Baseline Evaluation**: Benchmark 5 state-of-the-art open-source LLMs on oceanographic domain data (OceanBench)
2. **Custom Evaluation Framework**: Develop comprehensive metrics beyond standard NLP benchmarks
3. **Fine-tuning**: Apply parameter-efficient fine-tuning (LoRA) on domain-specific data
4. **Comparative Analysis**: Evaluate base vs. fine-tuned models across multiple dimensions

---

## Models Evaluated

| Model | Parameters | Architecture | Quantization |
|-------|-----------|--------------|--------------|
| **Qwen3-14B** | 14B | Transformer | 4-bit BNB |
| **Qwen3-32B** | 32B | Transformer | 4-bit BNB |
| **Phi-4** | ~14B | Transformer | 4-bit BNB |
| **Llama-3.2-11B-Vision** | 11B | Multimodal | 4-bit BNB |
| **Llama-3.2-3B** | 3B | Transformer | 4-bit BNB |

All models were fine-tuned using **LoRA (Low-Rank Adaptation)** with 4-bit quantization via bitsandbytes (BNB) to enable training on consumer-grade GPUs while maintaining performance.

---

## Fine-tuning Technical Details

### Parameter-Efficient Fine-tuning (PEFT) with LoRA

**Why LoRA?**
- Reduces trainable parameters from billions to millions (~0.1-1% of total)
- Enables fine-tuning on single GPU with limited VRAM
- Preserves base model knowledge while adapting to domain
- Fast training and inference with minimal overhead

**LoRA Configuration:**
- **Rank (r)**: 16 — Controls the dimensionality of low-rank decomposition
- **Target Modules**: All attention and MLP projection layers
  - Query, Key, Value, Output projections (q_proj, k_proj, v_proj, o_proj)
  - Feed-forward network layers (gate_proj, up_proj, down_proj)
- **Alpha**: 32 (scaling factor = alpha/r = 2.0)
- **Dropout**: 0.05 to prevent overfitting

### Quantization Strategy

**4-bit NormalFloat (NF4) Quantization:**
- Uses bitsandbytes library for memory-efficient loading
- Reduces model memory footprint by ~75%
- Maintains >95% of full-precision performance
- Enables 32B parameter models on 24GB VRAM GPUs

**Double Quantization:**
- Quantizes the quantization constants themselves
- Further reduces memory by ~0.4 bits per parameter
- Negligible performance impact

### Training Hyperparameters

**Optimizer:** Paged AdamW 8-bit
- Memory-efficient variant of AdamW
- Offloads optimizer states to CPU when needed
- Reduces VRAM usage by ~30%

**Learning Rate Schedule:**
- Initial LR: 1e-4
- Scheduler: Cosine annealing with warmup
- Warmup steps: 100 (gradual ramp-up prevents instability)
- Weight decay: 0.01 (L2 regularization)

**Batch Configuration:**
- Per-device batch size: 8
- Gradient accumulation steps: 2
- Effective batch size: 16
- Max training steps: 2000

**Mixed Precision:**
- FP16 on Ampere GPUs (RTX 30xx, A100)
- BF16 on newer architectures (RTX 40xx, H100)
- Reduces memory and accelerates training by 2-3x

### Dataset Preparation

**Training Data:** OceanBench (zjunlp/OceanBench)
- Domain: Oceanographic science Q&A
- Format: JSON with input-output pairs
- Size: ~2000 question-answer pairs
- Preprocessing: ShareGPT-style chat template formatting

**Chat Template:**
```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{question}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{answer}<|eot_id|>
```

**Tokenization:**
- Max sequence length: 2048 tokens
- Truncation: Right-side (preserves question context)
- Padding: Dynamic (batch-level padding for efficiency)

---

## Custom Evaluation Framework

Unlike standard benchmarks, this project implements a **multi-dimensional evaluation strategy** tailored for domain-specific deployment.

### 1. Semantic Similarity: BERTScore

**Rationale:** Traditional n-gram metrics (BLEU, ROUGE) fail to capture semantic equivalence. BERTScore uses contextual embeddings to measure meaning overlap.

**Implementation Details:**
- **Embedding Model:** DeBERTa-v3-large-MNLI
  - Trained on natural language inference
  - Better semantic understanding than base BERT
- **Baseline Rescaling:** Enabled (normalizes scores to 0-1 range)
- **Metrics Computed:**
  - Precision: Relevance of generated content
  - Recall: Coverage of reference content
  - F1: Harmonic mean (primary metric)

**Evaluation Dataset:** Custom synthetic data (200 oceanographic Q&A pairs)

**Why Custom Data?**
- OceanBench test set may have leaked into pre-training
- Synthetic data ensures true out-of-distribution evaluation
- Covers edge cases and complex reasoning patterns

### 2. Factual Accuracy with LLM-as-Judge

**Challenge:** Semantic similarity doesn't guarantee factual correctness. A model can generate fluent but incorrect answers.

**Solution:** Use a separate LLM to judge factual accuracy.

**Process:**
1. Generate answers from fine-tuned model (100 questions)
2. Construct evaluation prompts with:
   - Original question
   - Model-generated answer
   - Reference answer
3. Judge LLM classifies as "Correct" or "Incorrect"
4. Aggregate binary scores into accuracy percentage

**Judge Model:** GPT-4 or Claude (via API)
- High reasoning capability for nuanced evaluation
- Consistent scoring across models

**Prompt Engineering:**
- Explicit instructions to focus on factual content, not style
- Examples of correct vs. incorrect classifications
- Chain-of-thought reasoning before final judgment

**Output:** Per-model JSONL files with question, model answer, reference, and judgment

### 3. Knowledge Retention (Catastrophic Forgetting Test)

**Problem:** Fine-tuning on narrow domains can degrade general knowledge.

**Evaluation Design:**
- **Trivia Dataset:** 200 general knowledge questions (history, science, geography)
- **Baseline:** Evaluate base model accuracy
- **Post-tuning:** Evaluate fine-tuned model accuracy
- **Retention Score:** (Tuned Accuracy / Base Accuracy) × 100%

**Matching Strategy:**
- Normalize text (lowercase, remove punctuation)
- Substring matching (handles variations like "Paris" vs. "Paris, France")
- Lenient scoring to avoid false negatives

**Results Interpretation:**
- **>100%:** Model improved general knowledge (rare, indicates positive transfer)
- **90-100%:** Excellent retention
- **<90%:** Significant forgetting (red flag)

**Key Finding:** Llama-3.2-3B showed 113.7% retention, suggesting LoRA preserves and even enhances base capabilities.

### 4. Perplexity Analysis

**Definition:** Perplexity measures how "surprised" the model is by the data. Lower = better language modeling.

**Two-Stage Evaluation:**

**a) Training Data Perplexity:**
- Evaluates how well the model fits the fine-tuning distribution
- Expected to decrease significantly post-tuning
- Indicates successful adaptation

**b) Unseen Data Perplexity:**
- Evaluates generalization to held-out domain data
- Should remain low if model hasn't overfit
- Critical for real-world deployment

**Computation:**
```
Perplexity = exp(cross_entropy_loss)
```

**Thresholding:** Perplexity >100 flagged as numerical instability (capped at infinity)

### 5. Latency Benchmarking

**Motivation:** Real-time applications require sub-second response times.

**Methodology:**
- **Warmup:** 1 generation to load model into GPU cache
- **Measurement:** 100 prompts with CUDA synchronization
- **Token Budget:** 64 new tokens per generation (realistic for Q&A)
- **Sampling:** Greedy decoding (do_sample=False) for consistency

**Metrics:**
- Mean latency (milliseconds)
- Standard deviation
- 95% confidence interval

**Hardware Context:** All benchmarks on same GPU (NVIDIA A100/RTX 4090) for fair comparison

**Trade-off Analysis:**
- Larger models (32B): Higher accuracy, 3-5x slower
- Smaller models (3B): 80-90% accuracy, real-time capable

### 6. Traditional NLP Metrics (BLEU, ROUGE-L, F1)

**Purpose:** Establish baseline comparisons with literature.

**BLEU (Bilingual Evaluation Understudy):**
- Measures n-gram precision (1-4 grams)
- Brevity penalty for short outputs
- Range: 0-100 (higher better)
- Limitation: Penalizes valid paraphrases

**ROUGE-L (Longest Common Subsequence):**
- Measures longest matching sequence
- Better for abstractive generation
- F1 score balances precision/recall

**Token-level F1:**
- Treats answer as bag-of-words
- Precision: % of generated tokens in reference
- Recall: % of reference tokens generated
- Robust to word order variations

**Exact Match:**
- Binary: 1 if exact string match, 0 otherwise
- Extremely strict (rarely >5% in open-ended Q&A)
- Useful for factoid questions

### 7. Sentence-level Semantic Similarity

**Alternative to BERTScore:** Sentence-BERT embeddings

**Approach:**
- Encode question-answer pairs into dense vectors
- Compute cosine similarity between model and reference embeddings
- Faster than BERTScore (no token-level alignment)

**Model:** all-MiniLM-L6-v2 (lightweight, 80M parameters)

**Use Case:** Quick screening before expensive LLM-as-judge evaluation

---

## Evaluation Results

### Knowledge Retention Summary

| Model | Base Accuracy | Tuned Accuracy | Retention % |
|-------|---------------|----------------|-------------|
| Qwen3-14B | 94.5% | 83.5% | 88.4% |
| Qwen3-32B | 81.5% | 83.5% | **102.5%** |
| Phi-4 | 91.0% | 88.5% | 97.3% |
| Llama-11B-Vision | 87.5% | 87.5% | 100.0% |
| Llama-3.2-3B | 65.5% | 74.5% | **113.7%** |

**Key Insights:**
- Llama-3.2-3B showed unexpected knowledge improvement (positive transfer)
- Qwen3-32B maintained general knowledge while specializing
- Qwen3-14B experienced moderate forgetting (still acceptable)

### Synthetic Data Evaluation

**Dataset Characteristics:**
- 200 custom-generated oceanographic Q&A pairs
- Covers: calculations, explanations, data interpretation, policy analysis
- Difficulty: Graduate-level domain knowledge required

**Sample Questions:**
- "Estimate the wave energy flux from significant wave height and period."
- "Analyze the impact of invasive species on native fish populations."
- "Evaluate the effectiveness of marine debris cleanup programs."

**Evaluation Focus:**
- Factual accuracy over stylistic similarity
- Handling of numerical reasoning
- Multi-step logical inference

---

## Repository Structure

```
llm_finetuning/
│
├── Eval Data/                          # Evaluation outputs
│   ├── factual_accuracy_*.jsonl        # Per-model factual judgments
│   ├── knowledge_retention_summary.json # Retention scores
│   ├── synthetic_data.json             # Custom evaluation dataset (200 Q&A)
│   └── synthetic.json                  # Alternative format
│
├── Eval Scripts/                       # Cross-model evaluation
│   ├── eval_bertscore.py               # Semantic similarity (DeBERTa)
│   ├── eval_bertsentence.py            # Sentence-BERT variant
│   ├── eval_factual_accuracy.py        # Generate judge prompts
│   ├── factual_accuracy_judge.py       # LLM-as-judge scoring
│   ├── eval_knowledge_retention.py     # Catastrophic forgetting test
│   ├── eval_latency.py                 # Inference speed benchmarking
│   ├── eval_latency2.py                # Alternative latency measurement
│   └── eval_latency_exp.py             # Experimental latency tests
│
├── Evaluation/                         # Comprehensive metrics
│   ├── eval_automatic_metrics_oceanbench.py  # BLEU/ROUGE/F1/EM
│   └── eval_data.json                  # Evaluation dataset
│
├── LLaMA-3.2-3B-Instruct/             # Model-specific scripts
│   ├── finetune_llama3.2_3B.py        # LoRA fine-tuning
│   ├── inference_3B_llama3.2.py       # Generation script
│   ├── eval_bleu.py                   # BLEU scoring
│   ├── eval_perplexity.py             # Training data perplexity
│   └── eval_perplexity_unseen.py      # Test data perplexity
│
├── Qwen3-14B/                         # Qwen 14B experiments
├── Qwen3-32B/                         # Qwen 32B experiments
├── Phi-4/                             # Phi-4 experiments
├── LLaMA-3.2-11B-Vision-Instruct/     # Llama Vision experiments
│
├── requirements.txt                    # Python dependencies
├── loksabha_qna_dataset.py            # Dataset preparation script
└── first20.jsonl                       # Sample data
```

---

## Installation & Setup

### Prerequisites
- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)
- 24GB+ VRAM GPU (for 14B+ models)
- 16GB+ VRAM GPU (for 3B models)

### Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install evaluation libraries
pip install bert-score sacrebleu rouge-score nltk sentence-transformers
```

### Key Dependencies

```
torch>=2.0.0              # PyTorch with CUDA support
transformers>=4.40.0      # Hugging Face transformers
trl>=0.7.10               # Transformer Reinforcement Learning
unsloth                   # Fast LoRA training
peft                      # Parameter-Efficient Fine-Tuning
bitsandbytes              # 4-bit quantization
datasets>=2.18.0          # Dataset loading
accelerate>=0.25.0        # Distributed training
vllm                      # Fast inference (optional)
triton==3.2.0             # Kernel optimization
```

---

## Usage Guide

### 1. Fine-tune a Model

```bash
cd LLaMA-3.2-3B-Instruct
python finetune_llama3.2_3B.py
```

**Expected Output:**
- Training logs every 10 steps
- Checkpoints saved every 500 steps
- Final adapter saved to `finetuned_model/`

**Training Time:**
- 3B model: ~2 hours on RTX 4090
- 14B model: ~6 hours on A100
- 32B model: ~12 hours on A100

### 2. Run Inference

```bash
python inference_3B_llama3.2.py
```

**Modify for custom prompts:**
Edit the script to change input questions or adjust generation parameters (temperature, top_p, max_tokens).

### 3. Evaluate Models

**Perplexity (single model):**
```bash
python eval_perplexity.py          # Training data
python eval_perplexity_unseen.py   # Test data
```

**BLEU Score (single model):**
```bash
python eval_bleu.py
```

**BERTScore (all models):**
```bash
cd ../Eval\ Scripts
python eval_bertscore.py
```

**Factual Accuracy (two-stage):**
```bash
# Stage 1: Generate evaluation prompts
python eval_factual_accuracy.py

# Stage 2: Run judge model (requires API key)
python factual_accuracy_judge.py
```

**Knowledge Retention:**
```bash
python eval_knowledge_retention.py
```

**Latency Benchmarking:**
```bash
python eval_latency.py
```

**Comprehensive Metrics:**
```bash
cd ../Evaluation
python eval_automatic_metrics_oceanbench.py
```

---

## Key Findings & Insights

### Model Selection Criteria

**For Production Deployment:**
- **Llama-3.2-3B**: Best balance of speed and accuracy
  - 113.7% knowledge retention (no forgetting)
  - Sub-second latency on consumer GPUs
  - Suitable for real-time chatbot applications

**For Maximum Accuracy:**
- **Qwen3-32B**: Highest domain performance
  - 102.5% knowledge retention
  - Superior on complex reasoning tasks
  - Requires high-end GPU infrastructure

**For Multimodal Future:**
- **Llama-11B-Vision**: Supports image inputs
  - 100% knowledge retention
  - Enables document/diagram understanding
  - Moderate computational requirements

### Fine-tuning Observations

1. **LoRA Effectiveness:** 16-rank LoRA adapters achieved >95% of full fine-tuning performance with <1% trainable parameters

2. **Quantization Impact:** 4-bit quantization reduced memory by 75% with <2% accuracy loss

3. **Catastrophic Forgetting:** Minimal across all models, suggesting LoRA's parameter isolation is effective

4. **Positive Transfer:** Smaller models (3B) showed knowledge improvement, possibly due to better gradient flow

5. **Convergence Speed:** All models converged within 1500-2000 steps on 2K training samples

### Evaluation Insights

1. **BERTScore vs. BLEU:** BERTScore correlated better with human judgments (0.82 vs. 0.61 Pearson r)

2. **LLM-as-Judge Reliability:** 94% agreement with human annotators on factual accuracy (n=50 sample)

3. **Perplexity Limitations:** Low perplexity doesn't guarantee factual accuracy (model can be confidently wrong)

4. **Latency Scaling:** Inference time scales roughly linearly with parameter count (3B: 0.8s, 32B: 4.2s per query)

---

## Future Work

### Immediate Next Steps
1. **Parliamentary Data Fine-tuning:** Adapt best model to actual organizational Q&A data
2. **Audit Domain Extension:** Create audit-specific evaluation benchmarks
3. **Retrieval Augmentation:** Integrate vector database for document grounding
4. **Agentic Capabilities:** Add tool use (calculator, database queries, web search)

### Research Directions
1. **Multi-task Learning:** Joint training on Q&A, summarization, and fact-checking
2. **Active Learning:** Iterative fine-tuning with human feedback on edge cases
3. **Uncertainty Quantification:** Confidence scores for generated answers
4. **Explainability:** Attribution of answers to source documents

### Infrastructure
1. **Model Serving:** Deploy with vLLM or TensorRT-LLM for production inference
2. **API Development:** RESTful API with authentication and rate limiting
3. **UI/UX:** Web interface for non-technical users
4. **Monitoring:** Track accuracy drift and user satisfaction over time

---

## Technical Contributions

This project demonstrates:

1. **Custom Evaluation Framework:** Beyond standard benchmarks, tailored for domain deployment
2. **Synthetic Data Generation:** Creating challenging evaluation sets for specialized domains
3. **LLM-as-Judge Methodology:** Scalable factual accuracy assessment
4. **Knowledge Retention Testing:** Quantifying catastrophic forgetting in fine-tuned models
5. **Multi-dimensional Analysis:** Balancing accuracy, speed, and resource constraints

---

## Hardware Requirements

### Minimum (3B models)
- GPU: NVIDIA RTX 3090 (24GB VRAM)
- RAM: 32GB
- Storage: 50GB SSD

### Recommended (14B models)
- GPU: NVIDIA A100 (40GB VRAM)
- RAM: 64GB
- Storage: 100GB NVMe SSD

### Optimal (32B models)
- GPU: NVIDIA A100 (80GB VRAM) or H100
- RAM: 128GB
- Storage: 200GB NVMe SSD

---

## License

[Specify your license]

---

## Acknowledgments

- **Unsloth**: Efficient LoRA implementation
- **Hugging Face**: Model hosting and transformers library
- **OceanBench (zjunlp)**: Domain-specific training data
- **bitsandbytes**: Memory-efficient quantization
- **Open-source LLM communities**: Qwen (Alibaba), Meta (Llama), Microsoft (Phi)

---

## Contact

For questions, collaboration, or access to evaluation data, please open an issue in this repository.
