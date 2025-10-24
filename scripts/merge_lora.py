from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("./models/gemma-3-1b-it", torch_dtype="auto")
lora = PeftModel.from_pretrained(base, "./gemma-prescription-finetuned-it")

lora = lora.merge_and_unload()
lora.save_pretrained("./gemma-prescription-finetuned-it-merged")

tokenizer = AutoTokenizer.from_pretrained("./models/gemma-3-1b-it")
tokenizer.save_pretrained("./gemma-prescription-finetuned-it-merged")
