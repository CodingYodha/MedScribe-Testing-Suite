import torch
import json 
import pandas as pd
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset, load_from_disk


#Model Configurations...
MODEL_NAME = "./models/gemma-3-1b-it"
MAX_SEQ_LENGTH = 1024   
LOAD_IN_4BIT = True

#Training Configurations...
NUM_EPOCHS = 3
OUTPUT_DIR = "./gemma-prescription-finetuned-it"
BATCH_SIZE = 1
LEARNING_RATE = 2e-4
GRADIENT_ACCUMULATION_STEPS = 4 #helps in finetuning effectively as vram is less
#effective batch size = BATCH_SIZE*GRADIENT_ACCUMULATION_STEPS = 4*2 = 8
WARMUP_STEPS = 5  #starts lr from zero to 2e-4 within 5 steps , acts as a ramp
LOGGING_STEPS = 5 #logs the log 
SAVE_STEPS = 50 

#stps per epoch = No. of Training samples / (Batch size * Gradient accumulation steps)

#Lora Settings (efficient fine-tuning)
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
#while we use unsloth, but we still declare Lora config bcz,
#Lora controls parameter - efficient finetuning at method level
#Unsloth optimizes and accelerates training infrastrucutre and memory usage for the best performance

# DATASET_PATH ="/mnt/e/Projects/Med_Scribe/Medscribe_testing/prescription_finetuning_dataset_fixed.csv"




#===========defining stopping criteria===================
from transformers import StoppingCriteria, StoppingCriteriaList

class StopOnAAA(StoppingCriteria):
    def __init__(self, tokenizer):
        self.aaa_token_ids = tokenizer.encode("AAA", add_special_tokens=False)

    def __call__(self, input_ids, scores, **kwargs):
        # check if the last few generated tokens match "AAA"
        if len(input_ids[0]) >= len(self.aaa_token_ids):
            if input_ids[0][-len(self.aaa_token_ids):].tolist() == self.aaa_token_ids:
                return True
        return False

#========Model loading===========
print("Loading Model....")

model , tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    load_in_4bit = LOAD_IN_4BIT,
    dtype = None,
    max_seq_length=MAX_SEQ_LENGTH
)

print("Model Loaded...")


#==========Lora adapters (for efficiency)============

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj" , "down_proj",
    ],
    lora_alpha=LORA_ALPHA,
    lora_dropout = LORA_DROPOUT,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state=42,
    use_rslora=False,
    loftq_config=None,
)
print("Lora adapters added..")

#=========Loading & Preparing dataset=============

# print("loading the data...")

# df = pd.read_csv(DATASET_PATH)
# print(f"Loaded {len(df)} samples")

# #validating the correct json
# print("validating the dataset...")
# valid_samples = []
# invalid_count = 0

# for idx, row in df.iterrows():
#     try:
#         json.loads(row['output'])
#         valid_samples.append(row)
#     except json.JSONDecodeError:
#         invalid_count+=1
#         print("invalid JSON row {idx}")

# print(f"validated the samples , good samples: {len(valid_samples)}")
# if invalid_count>0:
#     print(f"invalid count: {invalid_count}, these are skipped")

# df_valid = pd.DataFrame(valid_samples)


#=========Load Pretokenized Dataset=============
from datasets import load_from_disk

print("Loading pretokenized datasets...")
train_dataset = load_from_disk("./tokenized_dataset_claude_with_instructions/train")
val_dataset = load_from_disk("./tokenized_dataset_claude_with_instructions/val")

print(f"Loaded pretokenized datasets: {len(train_dataset)} train, {len(val_dataset)} val")

#=============Setup Trainer============

print("\n Disabled Multiprocessing statements")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["HF_DATASETS_DISABLE_MULTIPROCESSING"] = "1"

import datasets
datasets.disable_caching()  
datasets.config.HF_DATASETS_DISABLE_MULTIPROCESSING = True


# Patch Unsloth to disable multiprocessing completely


print("\n Setting up trainer")
trainer = SFTTrainer(
    #loading model and config and output settings
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset = val_dataset,
    dataset_text_field=None,
    max_seq_length = MAX_SEQ_LENGTH,
    dataset_num_proc=1,
    packing=False,
    args=TrainingArguments(
        output_dir = OUTPUT_DIR,
        overwrite_output_dir = True,

        #training settings
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size = BATCH_SIZE,
        gradient_accumulation_steps= GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs = NUM_EPOCHS,
        learning_rate = LEARNING_RATE,

        #optimization
        fp16=not torch.cuda.is_bf16_supported(),
        optim="adamw_8bit" , #memory efficient optimizer
        weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_steps = WARMUP_STEPS,

        #logging and saving
        logging_steps = LOGGING_STEPS,
        save_steps = SAVE_STEPS,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",

        #Memory optimization
        gradient_checkpointing = True,

        #Reproducibility
        seed=42,

        #reporting
        report_to="none",

    ),
)

print("Trainer configured...........")

#========Train the Model=======
print("\n" + "="*80)
print("STARTING TRAINING")
print("="*80)
import torch, os

print("GPU:", torch.cuda.get_device_name(0))
print("Free VRAM:", torch.cuda.mem_get_info()[0] / 1e9, "GB")
os.environ["PYTHONWARNINGS"] = "ignore"

if torch.cuda.is_available():
    print(f"\nGPU Memory before training:")
    print(f"  Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print(f"  Reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")

#train!!!!!!
trainer_stats = trainer.train()
print("\n" + "="*80)
print("TRAINING COMPLETED")
print("="*80)
print(f"Training loss: {trainer_stats.training_loss:.4f}")
print(f"Training time: {trainer_stats.metrics['train_runtime']:.2f} seconds")
print("emptying cache")
torch.cuda.empty_cache()

#====================saving the model===========
print("\n Saving fine-tuned model...")

#saveing LoRA adapters only (lightweight)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Model saved to {OUTPUT_DIR}")

try:
    print("\nSaving merged model (for deployment)...")
    model.save_pretrained_merged(
        f"{OUTPUT_DIR}_merged",
        tokenizer,
        save_method="merged_16bit",
    )
    print(f"Merged model saved to: {OUTPUT_DIR}_merged")
except Exception as e:
    print(f"\nSkipping merged model save due to Unsloth bug: {e}")


#==========Evaluation=============
print("\n" + "="*80)
print("EVALUATING ON VALIDATION SET")
print("="*80)

eval_results = trainer.evaluate()

print(f"\nValidation Results:")
print(f"  Loss: {eval_results['eval_loss']:.4f}")
print(f"  Perplexity: {torch.exp(torch.tensor(eval_results['eval_loss'])):.2f}")


#=========Test Inference==========
print("\n" + "="*80)
print("TESTING INFERENCE")
print("="*80)

#enable fast inference mode
FastLanguageModel.for_inference(model)

test_conversation = """Mr. Shah, your blood pressure is quite high at 160/100. I'm starting you on Amlodipine 5mg once daily in the morning. Also take Atorvastatin 10mg at bedtime for your cholesterol. Get your lipid profile and kidney function tests done after 1 month. Reduce salt intake and exercise regularly."""

#Formatting input
test_input = f"""<start_of_turn>user
Extract medical entities from the following prescription conversation and output ONLY a valid JSON with this structure:
{{"medicines": [{{"name": str, "dosage": str, "frequency": str, "duration": str, "route": str, "timing": str}}], "diseases": [str], "symptoms": [str], "tests": [{{"name": str, "timing": str}}], "instructions": [str]}}

Conversation:
{test_conversation}<end_of_turn>
<start_of_turn>model
"""

#tokenize
inputs = tokenizer([test_input], return_tensors="pt").to(model.device)
stopping_criteria = StoppingCriteriaList([StopOnAAA(tokenizer)])
#generate
print("Generating Prediction....")
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.1,
    do_sample=False,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id = tokenizer.eos_token_id,
    stopping_criteria = stopping_criteria
)

input_len = inputs.input_ids.shape[1]
generated_tokens = outputs[0][input_len:]
prediction = tokenizer.decode(generated_tokens, skip_special_tokens=True)
prediction = prediction.replace("<start_of_turn>user", "").replace("<start_of_turn>model", "")
prediction = prediction.split("AAA")[0].strip()



print("\nTest Input:")
print(test_conversation)
print("\nModel Output:")
print(prediction)

import  json
import regex as re

def extract_json(text):
    # Remove code fences and stray language markers
    text = re.sub(r"```[a-z]*", "", text).strip()
    # Extract first {...} block
    match = re.search(r"\{(?:[^{}]|(?R))*\}", text)
    if not match:
        return None
    raw_json = match.group(0)
    # Minor bracket fixes
    raw_json = raw_json.replace(",{", ",").replace("},", "},")
    try:
        return json.loads(raw_json)
    except json.JSONDecodeError:
        # Fallback: try cleaning repeated braces and re-parse
        raw_json = re.sub(r"}\s*{", ",", raw_json)
        try:
            return json.loads(raw_json)
        except:
            return None


prediction = prediction.split("AAA")[0].strip()
fixed = extract_json(prediction)

if fixed:
    print("✓ Valid JSON:\n", json.dumps(fixed, indent=2))
else:
    print("✗ Could not repair JSON\n", prediction[:500])

# Validate JSON
# try:
#     parsed = json.loads(fixed)
#     print("\n✓ Valid JSON output!")
#     print(json.dumps(parsed, indent=2))
# except json.JSONDecodeError as e:
#     print(f"\n✗ Invalid JSON: {e}")

prediction = fixed

if torch.cuda.is_available():
    print("\n" + "="*80)
    print("MEMORY SUMMARY")
    print("="*80)
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
    print(f"Current GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

print(f"Training on {len(train_dataset)} samples | Validation on {len(val_dataset)} samples")
print(f"VRAM available: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")


print("\n" + "="*80)
print("FINE-TUNING COMPLETE!")
print("="*80)
print(f"\nModel saved at: {OUTPUT_DIR}")
print(f"Merged model at: {OUTPUT_DIR}_merged")
print("\nTo use the fine-tuned model for inference:")
print(f"  model, tokenizer = FastLanguageModel.from_pretrained('{OUTPUT_DIR}')")