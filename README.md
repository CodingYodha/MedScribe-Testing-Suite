# MedScribe-Testing-Suite

**MedScribe-Testing-Suite** is a research and development workspace for fine-tuning, evaluating, and benchmarking medical text understanding models—particularly focused on **prescription parsing** and **clinical dialogue extraction**.

This repository includes experiments, datasets, and training pipelines developed as part of the **MedScribe project**.

<p align="center">
  <img src="https://img.shields.io/badge/Model-RxStruct--Gemma--1B-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Base%20Model-Gemma--3--1B--IT-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Framework-Unsloth-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-CC--BY--NC--2.0-yellow?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Quantized-GGUF-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Downloads-7+-brightgreen?style=for-the-badge" />
</p>

<p align="center">
  <b>Fine-tuned Model:</b> <a href="https://huggingface.co/Shiva7706/RxStruct-Gemma-1B">RxStruct-Gemma-1B</a> |
  <b>Quantized Version:</b> <a href="https://huggingface.co/mradermacher/RxStruct-Gemma-1B-GGUF">GGUF Release</a>
</p>
<br>
---

## 1. Project Overview

The project focuses on building and evaluating large language models (LLMs) capable of:

- Extracting structured medical data from doctor-style prescription conversations  
- Identifying medicines, dosages, tests, and clinical advice  
- Producing valid JSON outputs suitable for downstream systems  

The fine-tuned model is based on **Gemma-3-1B-IT** using the **Unsloth** framework for efficient **LoRA fine-tuning**.

---

## 2. Repository Structure

MedScribe-Testing-Suite/  <br>
├── audio/ # Test audio samples for ASR evaluation <br>
├── can_be_ignored/ # Scratch files and experimental scripts <br>
├── datasets/ <br>
│ ├── Synthetic_4_llms_csv/ # Multi-LLM synthetic datasets (CSV) <br>
│ └── Synthetic_Claude_Sonnet_4-5_jsonl/ # Claude-generated datasets (JSONL) <br>
│ ├── raw/ # Untokenized data <br>
│ └── processed/ # Tokenized datasets (train/val) <br>
├── env/ # Environment, dependency, and export files <br>
├── logs/ # Terminal and run logs (Linux + Windows) <br>
├── models/ <br>
│ ├── gemma-3-1b-it/ # Base pre-trained model <br>
│ ├── finetuned/ <br>
│ │ ├── gemma-prescription-finetuned/ # Initial fine-tune <br>
│ │ ├── gemma-prescription-finetuned-it/ # Improved instruction-tuned version <br>
│ │ └── gemma-prescription-finetuned-it-merged_final/ # Fully merged model for deployment <br>
│ ├── large-v3/ # Whisper ASR model (optional) <br>
│ └── pretrained_models_diarization/# Speaker diarization models (SpeechBrain) <br>
├── notebooks/ # Research and exploratory notebooks <br>
├── scripts/ # All functional training and data scripts <br>
│ ├── claude_dataset_factory.py # Claude dataset generation script <br>
│ ├── data_generation.py # Base dataset synthesis script <br>
│ ├── dataset_cleaner.py # Cleans and validates datasets <br>
│ ├── gemma_finetuning_unsloth.py # Fine-tuning pipeline <br>
│ ├── merge_lora.py # Merges LoRA adapter weights into full model <br>
│ ├── tokenize_prescription_dataset.py # Tokenization pipeline for datasets <br>
│ └── json_dataset_validator.py # Validates structure and JSON correctness <br>
├── unsloth_compiled_cache/ # Cached Unsloth kernel optimizations <br>
├── requirements.txt # Project dependencies <br>
└── README.md # Project documentation <br>



---

## 3. Model Summary

| Attribute | Description |
|------------|-------------|
| **Base Model** | Gemma-3-1B-IT |
| **Fine-tuning Framework** | Unsloth (2025.10.7) |
| **Method** | LoRA (Rank=8, α=16, Dropout=0.05) |
| **Precision** | bfloat16 |
| **Sequence Length** | 1024 tokens |
| **Stop Token** | “AAA” (for controlled end of output) |
| **Training Samples** | 166 (train) / 19 (validation) |
| **Validation Loss** | 0.2435 |
| **Validation Perplexity** | 1.28 |

---

## 4. Dataset Details

### `Synthetic_4_llms_csv/`
- Dataset generated using multiple LLMs (Gemma, Mistral, GPT, Claude).  
- CSV format with input and output columns.  
- Focused on varied prescription types, edge cases, and Indian medical context.

### `Synthetic_Claude_Sonnet_4-5_jsonl/`
- Dataset generated exclusively via **Claude 3.5 Sonnet API**.  
- JSONL format for consistent parsing.  
- Includes `"AAA"` stopping marker in all outputs.  
- Tokenized using `tokenize_prescription_dataset.py` and stored under `processed/`.

---

## 5. Fine-Tuning and Training

Fine-tuning is handled using the **Unsloth lightweight pipeline**.  
**Main script:** `scripts/gemma_finetuning_unsloth.py`

### Key Features
- Memory-optimized for GPUs under 6 GB VRAM  
- Automatic gradient checkpointing and offloading  
- Fast LoRA training (1.3% parameters updated)  

### Training Hardware

| Component | Specification |
|------------|---------------|
| **GPU** | NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM) |
| **CUDA Version** | 13.0 |
| **Driver Version** | 581.08 |
| **Peak VRAM Usage** | ~2.52 GB |
| **System RAM** | 16 GB (Dell G15) |
| **OS** | Linux (WSL2 Devcontainer) |
| **Framework Stack** | PyTorch 2.8.0 + CUDA 12.8 + Unsloth 2025.10.7 |

---

## 6. Model Artifacts

| Path | Description |
|------|--------------|
| `models/finetuned/gemma-prescription-finetuned/` | Early LoRA fine-tune |
| `models/finetuned/gemma-prescription-finetuned-it/` | Final instruction-tuned checkpoint |
| `models/finetuned/gemma-prescription-finetuned-it-merged_final/` | Fully merged model for deployment |
| `models/gemma-3-1b-it/` | Base model used for fine-tuning |


## Merged Model Creation
```bash
python scripts/merge_lora.py
```

## 7. Example Inference

**Input:**
```
Mr. Shah, your blood pressure is quite high at 160/100.
I'm starting you on Amlodipine 5mg once daily in the morning.
Also take Atorvastatin 10mg at bedtime for your cholesterol.
```

**Model Output:**
```json
{
  "medicines": [
    {"name": "Amlodipine", "dosage": "5mg", "frequency": "once daily", "route": "oral", "timing": "morning"},
    {"name": "Atorvastatin", "dosage": "10mg", "frequency": "at bedtime", "route": "oral"}
  ],
  "diseases": ["high blood pressure"],
  "tests": [],
  "instructions": ["reduce salt intake", "exercise regularly"]
}
```

## 8. Post-processing Pipeline

Output predictions are validated and corrected using:

- **json_dataset_validator.py** — Repairs minor JSON format issues
- **Regex-based extractor** — Isolates valid JSON objects from model outputs
- **Stop-token trimming** — Removes text after "AAA"
- **Sanitization filters** — Drops duplicates and irrelevant tokens

## 9. Dependencies

A minimal requirements file is provided:
```ini
torch==2.8.0+cu126
transformers==4.56.2
bitsandbytes==0.48.1
accelerate==1.11.0
unsloth==2025.10.7
unsloth_zoo==2025.10.8
scikit-learn==1.7.2
faster-whisper==1.2.0
speechbrain==1.0.3
soundfile==0.13.1
librosa==0.10.2.post1
llama_cpp_python==0.3.16
anthropic==0.42.0
ipywidgets==8.1.7
datasets==4.2.0
huggingface-hub==0.35.3
```

**Installation:**
```bash
pip install -r requirements.txt
```

## 10. Authors and Attribution

**Author:**

Shivaprasad B. Gowda  


**Project Context:**

This repository represents the testing and research suite for the MedScribe system.  
The main production-ready repository will be hosted separately.

## 11. License and Usage

This code and model are released for research and educational use only.  
Commercial or clinical deployment requires prior permission.  
All synthetic datasets are non-patient data and free from PII.
