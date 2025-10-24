import os
import json
import time
from anthropic import Anthropic

# ==============================
# CONFIGURATION
# ==============================
MODEL_NAME = "claude-sonnet-4-5-20250929"   # Claude 3.5 Sonnet (latest as of Oct 2024)
BATCHES = 15                                # how many batches to generate
SAMPLES_PER_BATCH = 15                      # samples per batch
OUTPUT_FILE = "clean_prescription_dataset.jsonl"

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# ==============================
# SYSTEM PROMPT (STRICT INSTRUCTIONS)
# ==============================
BASE_PROMPT = """You are a medical dataset generator.
Your task is to create fine-tuning data for a **medical prescription entity extraction model**.

FORMAT RULES (IMPORTANT):
- Output STRICTLY in JSON Lines format.
- Each line must be valid JSON: {"input": "<doctor conversation>", "output": "<json string> AAA"}
- Do NOT include code blocks, backticks, explanations, or extra text.
- Each line must be self-contained and parseable with json.loads().
- Use natural, Indian doctor-style monologue (no patient replies).
- JSON inside 'output' must follow this schema:
  {
    "medicines": [
      {"name": "string", "dosage": "string", "frequency": "string",
       "duration": "string", "route": "string", "timing": "string"}
    ],
    "diseases": ["string"],
    "symptoms": ["string"],
    "tests": [
      {"name": "string", "timing": "string"}
    ],
    "instructions": ["string"]
  }
- If a field is missing, write "unspecified".
- Append a single space + AAA after the closing brace of the JSON inside 'output'.
- Use lowercase except for medicine brand names.
- Each sample must be unique and medically realistic.

DO NOT OUTPUT ANYTHING OTHER THAN JSONL ROWS.
"""


# ==============================
# VALIDATION UTILITIES
# ==============================
def is_valid_jsonl_line(line: str) -> bool:
    """Checks if a line is valid JSON with required structure."""
    try:
        obj = json.loads(line)
        if not isinstance(obj, dict):
            return False
        if "input" not in obj or "output" not in obj:
            return False

        # Validate the JSON inside 'output'
        output_text = obj["output"].strip().replace(" AAA", "")
        j = json.loads(output_text)
        required_keys = {"medicines", "diseases", "symptoms", "tests", "instructions"}
        if set(j.keys()) != required_keys:
            return False

        # Append AAA if missing
        if not obj["output"].endswith("AAA"):
            obj["output"] = obj["output"].strip() + " AAA"
        return obj
    except Exception:
        return False


# ==============================
# MAIN GENERATION LOOP
# ==============================
def generate_batch(batch_num):
    """Generate one batch of dataset entries using Claude."""
    prompt = BASE_PROMPT + f"\n\nGenerate {SAMPLES_PER_BATCH} new samples (Batch {batch_num}).\n"
    print(f"\nGenerating batch {batch_num} ...")

    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=6000,
        temperature=0.2,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )

    text = response.content[0].text.strip()
    lines = [l for l in text.splitlines() if l.strip()]
    print(f"Received {len(lines)} lines from Claude.")
    return lines


# ==============================
# RUN ALL BATCHES
# ==============================
all_valid = []

for b in range(1, BATCHES + 1):
    lines = generate_batch(b)
    valid_count = 0

    for i, line in enumerate(lines, 1):
        res = is_valid_jsonl_line(line)
        if res:
            all_valid.append(res)
            valid_count += 1
        else:
            print(f"Invalid line in batch {b}, row {i}")

    print(f"Batch {b}: {valid_count} valid samples.")
    time.sleep(3)  # polite delay between API calls

# ==============================
# SAVE FINAL MERGED DATASET
# ==============================
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for row in all_valid:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"\nAll done! Saved {len(all_valid)} validated samples to {OUTPUT_FILE}")
print("You can now use this file directly with Hugging Face Datasets.")
