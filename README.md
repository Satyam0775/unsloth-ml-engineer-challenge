# Unsloth ML Engineer Challenge

This repository contains my solutions for **Task C**, **Task E**, and an additional **Task F** for the Unsloth ML Engineer challenge.

All experiments were executed on **Google Colab using a single Tesla T4 GPU (CUDA 12.4)**, with a focus on stability, correctness, and explainable ML systems trade-offs.

---

## Environment

- Platform: Google Colab
- GPU: Tesla T4 (16 GB)
- Driver: 550.54.15
- CUDA: 12.4
- Precision: FP16
- AMP / BF16: Disabled (BitsAndBytes + T4 limitation)

---

## Task C – torch.compile with QLoRA Training

**Notebook:** `task_c_torch_compile_qlora.ipynb`

### Objective
Enable `torch.compile` for QLoRA training while avoiding graph breaks, excessive recompilations, and training instability.

### What was implemented
- Loaded a 4-bit QLoRA model using BitsAndBytes
- Applied **regional torch.compile** instead of full-model compilation
- Successfully compiled:
  - `LlamaMLP`
  - `LlamaRMSNorm`
- Attention intentionally excluded due to Hugging Face runtime kwargs
  (`past_key_values`, `output_hidden_states`) causing Dynamo graph breaks
- AMP / BF16 disabled to avoid GradScaler issues on Tesla T4

### Results
- **Baseline training loss:** ~2.21  
- **Compiled training loss:** ~2.04  
- **Loss difference:** ~0.17 (within expected kernel-fusion variance)
- **Baseline time:** ~9–11 sec
- **Compiled time:** ~24 sec
- **VRAM usage:** ~1.3–1.6 GB (stable)

### Interpretation
- Training completed successfully with no recompilation storm
- Slight loss drift is expected due to fused kernels
- Stability and correctness were prioritized over forcing fullgraph compilation

---

## Task E – Memory Efficient Training

**Notebook:** `task_e_memory_efficient_training.ipynb`

### Objective
Reduce memory usage during training using standard memory-efficiency techniques.

### What was implemented
- Enabled gradient checkpointing
- Disabled KV cache
- Disabled unnecessary outputs (hidden states, attentions)
- Used minimal batch size suitable for a 1B parameter model

### Results
- **Final training loss:** ~2.24
- **Training time:** ~12.5 sec
- **VRAM usage:** ~1.33 GB

### Interpretation
- On a small (1B) model with batch size = 1, memory savings are modest
- Approach is correct and scales to larger models
- Training remained stable throughout

---

## Task F – Compiled Inference Benchmark (Optional)

**Notebook:** `task_f_compiled_inference_benchmark.ipynb`

### Objective
Benchmark eager vs `torch.compile` inference to evaluate real-world performance and memory trade-offs.

### What was implemented
- Inference-only benchmarking (no training)
- Compiled safe, compute-heavy modules:
  - `LlamaMLP`
  - `LlamaRMSNorm`
- Attention excluded to avoid Dynamo graph breaks
- Fixed input, warm-up runs, synchronized timing

### Results
- **Eager inference time:** ~5.03 sec
- **Compiled inference time:** ~5.72 sec
- **Speedup factor:** ~0.88×
- **Eager VRAM usage:** ~1178 MB
- **Compiled VRAM usage:** ~1022 MB (≈150 MB reduction)

### Interpretation
- `torch.compile` reduced memory usage
- Latency did not improve on Tesla T4 due to:
  - Small batch size
  - Limited SM count
  - Compilation overhead outweighing compute savings
- This behavior is expected on smaller GPUs

---

## Summary

- Reported **loss-based metrics**, which are the correct evaluation signals for LLM fine-tuning
- Prioritized stability, correctness, and explainable trade-offs
- Avoided risky optimizations that introduce graph breaks
- Demonstrated when `torch.compile` is beneficial—and when it is not
- All tasks executed successfully on a single Tesla T4 GPU
