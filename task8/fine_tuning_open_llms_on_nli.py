
import os
import re
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

from tqdm.auto import tqdm


# ==========================================
# 1. Configuration & Setup
# ==========================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    "./qwen_snli_output"
)

NUM_EPOCHS = 1

MAX_LENGTH = 256

TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2

LEARNING_RATE = 2e-4

EVAL_STEPS = 1000
SAVE_STEPS = 1000

SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

torch.manual_seed(SEED)

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is unavailable. Check the Kubernetes GPU allocation "
        "and PyTorch CUDA installation."
    )

device = torch.device("cuda")

# Select the best supported GPU precision.
use_bf16 = torch.cuda.is_bf16_supported()

model_dtype = (
    torch.bfloat16 if use_bf16 else torch.float16
)

print("\nGPU Information")
print("------------------------------")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: "
      f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print(f"BF16 supported: {use_bf16}")
print(f"Model dtype: {model_dtype}")


# ==========================================
# 2. Load Tokenizer and Model
# ==========================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=model_dtype,
)

model.config.use_cache = False


# ==========================================
# 3. LoRA Configuration
# ==========================================

print("\nApplying LoRA adapters...")

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

model = get_peft_model(
    model,
    peft_config
)

model.print_trainable_parameters()


# ==========================================
# 4. Load and Prepare SNLI Dataset
# ==========================================

print("\nLoading SNLI dataset...")

raw_datasets = load_dataset("stanfordnlp/snli")

# Remove invalid labels from every split.
raw_datasets = raw_datasets.filter(
    lambda example: example["label"] != -1
)

label_map = {
    0: "entailment",
    1: "neutral",
    2: "contradiction",
}

SYSTEM_PROMPT = (
    "You are a natural language inference classifier. "
    "Determine whether the hypothesis is entailed by, "
    "neutral toward, or contradicts the premise. "
    "Output exactly one word: entailment, neutral, "
    "or contradiction."
)


def format_dataset(example):

    prompt = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                f"Premise: {example['premise']}\n"
                f"Hypothesis: {example['hypothesis']}"
            ),
        },
    ]

    completion = [
        {
            "role": "assistant",
            "content": label_map[example["label"]],
        }
    ]

    return {
        "prompt": prompt,
        "completion": completion,
    }


print("Formatting training and validation datasets...")

formatted_datasets = raw_datasets.map(
    format_dataset,
    remove_columns=raw_datasets["train"].column_names,
)

train_dataset = formatted_datasets["train"]
eval_dataset = formatted_datasets["validation"]

print(f"Training examples: {len(train_dataset)}")
print(f"Validation examples: {len(eval_dataset)}")


# ==========================================
# 5. Configure SFTTrainer
# ==========================================

print("\nSetting up SFTTrainer...")

training_args = SFTConfig(

    output_dir=OUTPUT_DIR,

    # Training
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    max_grad_norm=1.0,

    # Sequence processing
    max_length=MAX_LENGTH,
    packing=False,
    completion_only_loss=True,

    # Mixed precision
    bf16=use_bf16,
    fp16=not use_bf16,

    # Memory optimization
    activation_checkpointing=True,

    # GPU optimization
    optim="adamw_torch_fused",
    dataloader_num_workers=2,
    dataloader_pin_memory=True,

    # Evaluation
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,

    # Checkpointing
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=2,

    # Logging
    logging_strategy="steps",
    logging_steps=25,
    logging_first_step=True,
    report_to="none",

    # Other
    seed=SEED,
    data_seed=SEED,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)


# ==========================================
# 6. Train
# ==========================================

print("\nStarting training...")

trainer.train()

print("\nTraining completed!")


# ==========================================
# 7. Save Final Model
# ==========================================

print("\nSaving final LoRA adapter...")

trainer.save_model(OUTPUT_DIR)

tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Model saved to: {OUTPUT_DIR}")


# ==========================================
# 8. Evaluate Classification Accuracy
# ==========================================

print("\nStarting classification evaluation...")

model = trainer.model
model.eval()

# Restore cache for inference.
model.config.use_cache = True

# Left padding is appropriate for batched generation.
tokenizer.padding_side = "left"

test_dataset = raw_datasets["test"]

correct = 0
total = 0
invalid_format = 0

EVAL_GENERATION_BATCH_SIZE = 32

for start in tqdm(
    range(0, len(test_dataset), EVAL_GENERATION_BATCH_SIZE),
    desc="Evaluating test set",
):

    batch = test_dataset[
        start:start + EVAL_GENERATION_BATCH_SIZE
    ]

    messages = []

    for premise, hypothesis in zip(
        batch["premise"],
        batch["hypothesis"],
    ):

        messages.append([
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"Premise: {premise}\n"
                    f"Hypothesis: {hypothesis}"
                ),
            },
        ])

    prompts = [
        tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )
        for message in messages
    ]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    ).to(device)

    with torch.inference_mode():

        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[
        :, inputs["input_ids"].shape[1]:
    ]

    predictions = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    for prediction, true_id in zip(
        predictions,
        batch["label"],
    ):

        prediction = prediction.lower().strip()

        match = re.fullmatch(
            r"\s*(entailment|neutral|contradiction)\s*[.!]?\s*",
            prediction,
        )

        if match:
            predicted_label = match.group(1)
        else:
            predicted_label = "invalid"
            invalid_format += 1

        true_label = label_map[true_id]

        if predicted_label == true_label:
            correct += 1

        total += 1


# ==========================================
# 9. Final Results
# ==========================================

accuracy = correct / total if total else 0.0

print("\nFinal Evaluation Results")
print("------------------------------")
print(f"Test samples: {total}")
print(f"Correct predictions: {correct}")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Invalid outputs: {invalid_format}")
print(f"Invalid output rate: "
      f"{invalid_format / total * 100:.2f}%" if total else
      "Invalid output rate: N/A")

print("\nFine-tuning experiment completed!")
