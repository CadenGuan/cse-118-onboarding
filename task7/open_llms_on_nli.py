import torch
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    DataCollatorForLanguageModeling,
    Trainer, 
    TrainingArguments
)

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
output_dir = "./qwen_snli_output"

# load the dataset
raw_datasets = load_dataset("stanfordnlp/snli")
raw_datasets = raw_datasets.filter(lambda x: x["label"] != -1)

raw_datasets["train"] = raw_datasets["train"].select(range(5000))
raw_datasets["validation"] = raw_datasets["validation"].select(range(500))


# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Qwen needs a pad token for batching
tokenizer.pad_token = tokenizer.eos_token


# Load Model in 4-bit (To save memory)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto" # auto FOR NAUTILUS
    )

# Format the data as a Chat and Tokenize
"""
For classification: 
def tokenize_function(example):
    tokenizer(example["premise"], example["hypothesis"], truncation=True)
"""

label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}

def format_and_tokenize(example):
    # Build the chat messages
    messages = [
        {"role": "system", "content": "Output exactly one word: entailment, neutral, or contradiction."},
        {"role": "user", "content": f"Premise: {example['premise']}\nHypothesis: {example['hypothesis']}"},
        {"role": "assistant", "content": label_map[example['label']]}
    ]
    # Apply Qwen's chat template
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    
    # Tokenize the final string
    tokenized = tokenizer(text, truncation=True, max_length=256, padding="max_length")
    
    # For Causal LM, labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_datasets = raw_datasets.map(format_and_tokenize) # Batched=True passes in a DICTIONARY of lists
tokenized_datasets.set_format("torch")

"""
For Classification: 
Dynamic padding to pad to the longest element in the batch
data_collector = DataCollatorWithPadding(tokenizer=tokenizer)
"""
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Define Trainer
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    eval_strategy="steps",
    eval_steps=50,                   
    logging_steps=10,
    save_total_limit=2,
    report_to="none",
    fp16=True # SET TO TRUE FOR NAUTILUS
)



"""
Evaluation
For Classification (Exact match, hard to achieve with generative language model)
def compute_metrics(eval_preds):
    metric = evaluate.load("stanfordnlp/snli") Doesn't work because evaluate.load expects a METRIC NAME
    logits, labels = eval_preds
    # compare the scores
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, reference=labels)
"""




# Setting up the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    # compute_metrics=compute_metrics,
    )

trainer.train()
trainer.save_model(output_dir)
