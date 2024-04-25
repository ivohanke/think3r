from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset, DatasetDict
import os

# Load dataset from disk if available, otherwise download it
dataset_path = "./dair-ai-emotion.hf"
if os.path.exists(dataset_path):
    datasets = DatasetDict.load_from_disk(dataset_path)
    print('loaded locally')
else:
    datasets = load_dataset("dair-ai/emotion")
    datasets.save_to_disk(dataset_path)

# Define the model and tokenizer paths
model_name = "microsoft/Phi-3-mini-128k-instruct"
cache_dir = './model_cache'

# Load model and tokenizer from local cache if available
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)

# Tokenizer and Data Collator setup
tokenizer.pad_token = tokenizer.eos_token  # Ensure compatibility with padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Tokenization function for processing the dataset
def tokenize_function(examples):
    result = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    result["labels"] = result["input_ids"].copy()  # Prepare labels for causal language modeling
    return result

# Apply tokenization and handle dataset columns
tokenized_datasets = datasets.map(tokenize_function, batched=True, remove_columns=["text"])
print(tokenized_datasets)

# Example prompt and response generation
prompt = "Tell me a joke"
inputs = tokenizer(prompt, return_tensors="pt")
generate_ids = model.generate(inputs.input_ids, max_length=300)
print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])