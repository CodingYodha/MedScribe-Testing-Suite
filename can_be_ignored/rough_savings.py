#===========Formatting data for training (prompts)=========
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
    
        "medicines": [
        {{
            
                "name": str, 
                "dosage": str, 
                "frequency": str, 
                "duration": str, 
                "route": str, 
                "timing": str
        }}
        
        ], 
        "diseases": [str], 
        "symptoms": [str], 
        "tests": [
        {{
            "name": str, 
            "timing": str
        }}
        
        ], 
        "instructions": [str]
    }}


Conversation:
{input}<end_of_turn>
<start_of_turn>model
{output}<end_of_turn>"""

def format_sample(sample):
    """Format each data entry into the following Gemma's chat template"""
    return INSTRUCTION_TEMPLATE.format(
        input = sample['input'],
        output = sample['output']
    )

#applying foramtting
df_valid['text'] = df_valid.apply(format_sample, axis=1)


train_size = int(0.9*len(df_valid))
train_df = df_valid[:train_size]
valid_df = df_valid[train_size:]

print(f"Training samples: {len(train_df)} ")
print(f"Validation samples: {valid_df}")
print("\nSample formatted training example:")
print("-" * 80)
print(train_df.iloc[0]['text'][:500] + "...")
print("-" * 80)

print("Converting to Hugginface dataset")
#Converting to HuggingFace dataset
#for lazyloading, RAM Efficiency
train_dataset = Dataset.from_pandas(train_df[['text']])
val_dataset= Dataset.from_pandas(valid_df[['text']])

print("\nSample formatted training example:")
print("-" * 80)
print(train_dataset[0]['text'][:500] + "...")
print("-" * 80)