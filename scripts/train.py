# scripts/train.py: Model fine-tuning

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from nltk.translate.bleu_score import sentence_bleu
import config

# Load dataset
dataset = load_dataset("json", data_files="data/combined_dataset.json", split="train")

# Tokenize function
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
def tokenize_function(examples):
    return tokenizer(examples["input"] + " " + examples["response"], truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Load model with LoRA
model = AutoModelForCausalLM.from_pretrained(config.MODEL_NAME)
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora_config)

# Training args
training_args = TrainingArguments(
    output_dir=config.MODEL_PATH,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=500,
)

# Trainer
trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset)
trainer.train()

# Save model
trainer.save_model(config.MODEL_PATH)

# Evaluation (BLEU score example)
def evaluate_bleu():
    # Simplified; use validation split in production
    reference = ["This is a test response."]
    candidate = model.generate(tokenizer.encode("Test input", return_tensors="pt"))[0]
    score = sentence_bleu(reference, tokenizer.decode(candidate))
    print(f"BLEU Score: {score}")

evaluate_bleu()
