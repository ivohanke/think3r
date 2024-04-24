from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset

# Potentail dataset to use (for classification of input)
datasets = load_dataset("dair-ai/emotion")

access_token = "" # Removed
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", token=access_token)


tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b", token=access_token)
tokenizer.pad_token = tokenizer.eos_token  # Ensure GPT-2 can handle padding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def tokenize_function(examples):
    # Tokenize the texts
    result = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    # GPT-2 uses the same input as labels for language modeling tasks.
    # Shift input ids to the right to create labels.
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_datasets = datasets.map(tokenize_function, batched=True, remove_columns=["text"])
print(tokenized_datasets)

prompt = "What is your favorite condiment?"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=30)
tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


 # Training and evaluation
training_args = TrainingArguments(
    output_dir="./data/gemma-2b",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=1,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"], 
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model("./data/model")

eval_results = trainer.evaluate()
print(eval_results)