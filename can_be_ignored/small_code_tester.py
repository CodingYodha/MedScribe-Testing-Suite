import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

# ===================================================
# CONFIG
# ===================================================
MODEL_NAME = "./models/gemma-3-1b-it"
DATASET_PATH = "/mnt/e/Projects/Med_Scribe/Medscribe_testing/prescription_finetuning_dataset_fixed.csv"
OUTPUT_DIR = "./tokenized_dataset_with_instructions"
MAX_SEQ_LENGTH = 1024

# ===================================================
# LOAD DATA
# ===================================================
df = pd.read_csv(DATASET_PATH)

# Just keep valid samples if needed
df = df.dropna(subset=["input", "output"])
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

df["text"] = df.apply(lambda r: INSTRUCTION_TEMPLATE.format(input=r["input"], output=r["output"]), axis=1)

print("\n\n")

dataset = Dataset.from_pandas(df[["text"]])
print(dataset[:1], "\n")
from datasets import load_from_disk
ds = load_from_disk("./tokenized_dataset_with_instructions/train")
print(ds[0]['text'][-200:])
