import csv, json

raw_path = "/mnt/e/Projects/Med_Scribe/Medscribe_testing/prescription_finetuning_dataset.csv"
fixed_path = "/mnt/e/Projects/Med_Scribe/Medscribe_testing/prescription_finetuning_dataset_fixed.csv"

rows = []
with open(raw_path, encoding="utf-8", errors="ignore") as f:
    buffer = []
    for line in f:
        line = line.strip()
        if not line:
            continue
        buffer.append(line)
        # Detect probable end of one record (JSON closing brace)
        if line.endswith("}") or line.endswith("}\""):
            text = " ".join(buffer)
            buffer.clear()
            # Split at the last JSON opening
            idx = text.find("{\"medicines")
            if idx != -1:
                input_part = text[:idx].strip().strip('"')
                output_part = text[idx:].strip().strip('"')
                rows.append([input_part, output_part])

# Write clean CSV
with open(fixed_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["input", "output"])
    writer.writerows(rows)

print(f"✅ Fixed CSV saved at: {fixed_path}")
print(f"Recovered {len(rows)} records.")
