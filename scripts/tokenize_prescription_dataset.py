import os
from datasets import load_dataset
from transformers import AutoTokenizer

# ===================================================
# CONFIG
# ===================================================
MODEL_NAME = "./models/gemma-3-1b-it"
DATASET_PATH = "clean_prescription_dataset.jsonl"
OUTPUT_DIR = "./tokenized_dataset_claude_with_instructions"
MAX_SEQ_LENGTH = 1024

# ===================================================
# LOAD DATA
# ===================================================
print("Loading dataset...")
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

print(f"Loaded {len(dataset)} samples from {DATASET_PATH}")

# ===================================================
# APPLY INSTRUCTION TEMPLATE
# ===================================================
INSTRUCTION_TEMPLATE = """<start_of_turn>user
You are a medical prescription parser. Extract ONLY information explicitly stated.

Rules:
1. Extract medicines with EXACT dosages mentioned
2. If dosage/frequency unclear, mark as "unspecified"
3. Do NOT infer or assume any information
4. If doctor says "continue previous meds", extract NOTHING
5. Output only one valid JSON object and stop
6. At the end of the Output print AAA
Output structure:
{{
  "medicines": [{{"name": str, "dosage": str, "frequency": str, "duration": str, "route": str, "timing": str}}],
  "diseases": [str],
  "symptoms": [str],
  "tests": [{{"name": str, "timing": str}}],
  "instructions": [str]
}}

Conversation:
{input}<end_of_turn>
<start_of_turn>model
{output}<end_of_turn>"""

def format_fn(example):
    return {
        "text": INSTRUCTION_TEMPLATE.format(input=example["input"], output=example["output"] + " AAA ")
    }

print("Formatting samples...")
dataset = dataset.map(format_fn, remove_columns=dataset.column_names)
print(f"Formatted dataset. Example:\n{dataset[0]['text'][:300]}...")

# ===================================================
# SPLIT TRAIN/VAL
# ===================================================
train_test = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_test["train"]
val_dataset = train_test["test"]

# ===================================================
# TOKENIZATION
# ===================================================
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
    )

print("Tokenizing dataset (single process)...")
tokenized_train = train_dataset.map(tokenize_fn, batched=True, num_proc=1)
tokenized_val = val_dataset.map(tokenize_fn, batched=True, num_proc=1)
tokenized_train = tokenized_train.remove_columns(["text"])
tokenized_val = tokenized_val.remove_columns(["text"])

# ===================================================
# SAVE
# ===================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
tokenized_train.save_to_disk(f"{OUTPUT_DIR}/train")
tokenized_val.save_to_disk(f"{OUTPUT_DIR}/val")

print(f"Tokenized datasets saved to {OUTPUT_DIR}")
print(f"Train: {len(tokenized_train)} | Val: {len(tokenized_val)}")
