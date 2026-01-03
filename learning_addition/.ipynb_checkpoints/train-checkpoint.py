import torch
from torch.utils.data import Dataset
from transformers import (
    GPT2Config, 
    GPT2LMHeadModel, 
    Trainer, 
    TrainingArguments, 
    default_data_collator
)

# 1. Manual Character Mapping
# We include digits, +, = and a space/padding token
chars = sorted(list("0123456789+="))
vocab = {char: i for i, char in enumerate(chars)}
vocab_size = len(chars)

class AdditionDataset(Dataset):
    def __init__(self, file_path, max_length=16):
        with open(file_path, 'r') as f:
            self.lines = [line.strip() for line in f]
        self.max_length = max_length

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        text = self.lines[idx]
        # Convert chars to IDs
        input_ids = [vocab[c] for c in text]
        
        # Padding: Use the '=' token as padding to fill up to max_length
        # This ensures all tensors in a batch are the same size
        padding_len = self.max_length - len(input_ids)
        input_ids += [vocab['=']] * padding_len
        
        # In causal LM training, labels are the same as input_ids
        # The model internally shifts them to predict the next token
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(input_ids, dtype=torch.long)
        }

# 2. Model Configuration
# We want a small model that fits easily on a T4 GPU
config = GPT2Config(
    vocab_size=vocab_size,
    n_positions=128,
    n_embd=256,
    n_layer=6,
    n_head=8,
    bos_token_id=vocab['='],
    eos_token_id=vocab['='],
    pad_token_id=vocab['='],
)
model = GPT2LMHeadModel(config)

# 3. Training Arguments
training_args = TrainingArguments(
    output_dir="./add_model",
    overwrite_output_dir=True,
    num_train_epochs=15,             # 10-20 epochs is usually enough for 10k samples
    per_device_train_batch_size=64,
    learning_rate=5e-4,
    weight_decay=0.01,
    eval_strategy="epoch",           # Uses the modern name for the strategy
    save_strategy="epoch",
    logging_steps=50,
    load_best_model_at_end=True,
    report_to="none",                # Prevents needing wandb/tensorboard logins
    fp16=torch.cuda.is_available(),  # Uses mixed precision if on GPU
)

# Initialize Datasets
train_ds = AdditionDataset("train.txt")
test_ds = AdditionDataset("test_id.txt")

# 4. Initialize Trainer with default_data_collator
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    data_collator=default_data_collator, # This is the key fix
)

# Start training
trainer.train()

# Save the final model
model.save_pretrained("./final_model")
print("Training complete! Model saved to ./final_model")
