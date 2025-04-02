import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
import numpy as np
import evaluate


# Load dataset
train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

# Reduce dataset size for faster training
train_data = train_data.sample(n=50000, random_state=42)
test_data = test_data.sample(n=10000, random_state=42)

print("Updated training samples:", len(train_data))
print("Updated test samples:", len(test_data))

# Convert labels to numerical values
label_map = {"Negative": 0, "Positive": 1}
train_data["label"] = train_data["label"].map(label_map)
test_data["label"] = test_data["label"].map(label_map)

train_data = train_data.dropna(subset=["label"])
test_data = test_data.dropna(subset=["label"])

print("Sample training data:\n", train_data.head())

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Convert dataset to PyTorch format
class SentimentDataset(Dataset):
    def __init__(self, data):
        self.texts = list(data["review"])
        self.labels = list(data["label"])

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(self.texts[idx], padding="max_length", truncation=True, max_length=512)
        
        return {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

# Create dataset
train_dataset = SentimentDataset(train_data)
test_dataset = SentimentDataset(test_data)

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Training arguments 
training_args = TrainingArguments(
    output_dir="./bert_model",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True
)

# Compute accuracy metric
def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Save trained model
model.save_pretrained("./bert_model")
tokenizer.save_pretrained("./bert_model")
print("Fine-tuning completed! Model saved.")
