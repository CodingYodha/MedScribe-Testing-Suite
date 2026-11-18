# MedScribe-Testing-Suite

Fine-tuning and testing a small LLM that reads a doctor–patient prescription conversation and returns structured JSON: medicines, dosage, frequency, tests and advice.

This is the research and testing workspace for the MedScribe project.

- Fine-tuned model: [RxStruct-Gemma-1B on Hugging Face](https://huggingface.co/Shiva7706/RxStruct-Gemma-1B)
- Quantized version: [RxStruct-Gemma-1B-GGUF](https://huggingface.co/mradermacher/RxStruct-Gemma-1B-GGUF)

## Problem

Doctors say prescriptions out loud during a consultation. Downstream systems need that information as clean, structured data, not free text. The goal is a model small enough to run on a laptop GPU that turns the conversation into valid JSON.

## How it works

1. **Synthetic data:** prescription-style conversations were generated with several LLMs, including the Claude API. Every output ends with an `AAA` stop marker. There are no real patients and no personal data.
2. **Cleaning and tokenizing:** scripts clean the data, check that every output is valid JSON, and tokenize it into train/validation splits.
3. **Fine-tuning:** Gemma-3-1B-IT is fine-tuned with LoRA using Unsloth, in bfloat16 with 1024-token sequences.
4. **Merging:** the LoRA adapter is merged into the base model for deployment.
5. **Post-processing:** a regex extractor pulls the JSON object out of the output, text after `AAA` is cut, and small JSON format issues are repaired.

The `audio/` folder and `scripts/diarize_and_label.py` are for testing the speech-to-text and speaker-labelling side of the pipeline (faster-whisper and SpeechBrain are in the requirements).

## Results

| | |
|---|---|
| Base model | Gemma-3-1B-IT |
| Method | LoRA (rank 8, alpha 16, dropout 0.05), about 1.3% of parameters trained |
| Data | 166 training / 19 validation samples |
| Validation loss | 0.2435 |
| Validation perplexity | 1.28 |
| Hardware | RTX 3050 Laptop GPU (6 GB), peak VRAM about 2.52 GB |

**Example**

Input:

```
Mr. Shah, your blood pressure is quite high at 160/100.
I'm starting you on Amlodipine 5mg once daily in the morning.
Also take Atorvastatin 10mg at bedtime for your cholesterol.
```

Model output:

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

## How to run

```bash
pip install -r requirements.txt

# fine-tune
python scripts/gemma_finetuning_unsloth.py

# merge the LoRA adapter into the base model
python scripts/merge_lora.py
```

Main pinned versions: torch 2.8.0, transformers 4.56.2, unsloth 2025.10.7. The full environment files are in `env/`.

## Repo layout

| Path | What it is |
|---|---|
| `scripts/` | Data generation, cleaning, validation, tokenizing, fine-tuning and LoRA merge scripts |
| `datasets/` | Synthetic datasets (CSV from several LLMs, JSONL from Claude) |
| `notebooks/` | Model loading, quantization and testing notebooks |
| `audio/` | Test audio for speech-to-text |
| `env/` | Environment and dependency exports |
| `logs/` | Training and terminal logs |
| `can_be_ignored/` | Scratch files |

## License

For research and educational use only. Commercial or clinical use needs permission first.
