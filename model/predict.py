import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import numpy as np

# Load fine-tuned model and tokenizer
print("Loading model and tokenizer...")
model_path = "./bert_model"  
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
model.eval()
print("Model loaded successfully!")

# Load test dataset 
print("Loading test data...")
test_data = pd.read_csv("data/test.csv").sample(n=5000, random_state=42)
label_map = {"Negative": 0, "Positive": 1}
test_data["label"] = test_data["label"].map(label_map)
print(f"Total test samples: {len(test_data)}")

# Function to predict sentiment
def predict_sentiment(text):
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        output = model(**encoding)
    prediction = torch.argmax(output.logits, dim=-1).item()
    return prediction

# Run predictions
print("Starting prediction loop...")
correct = 0
total = len(test_data)

for idx, (_, row) in enumerate(test_data.iterrows(), start=1):
    predicted_label = predict_sentiment(row["review"])
    true_label = row["label"]
    if predicted_label == true_label:
        correct += 1

    # Show progress every 10 samples
    if idx % 10 == 0 or idx == total:
        print(f"Progress: {idx}/{total} samples processed...")

# Calculate accuracy
accuracy = (correct / total) * 100
print(f"Accuracy on 5,000 test samples: {accuracy:.2f}%")
