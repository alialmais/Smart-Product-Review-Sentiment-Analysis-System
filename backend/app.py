from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from backend.db import products_collection, reviews_collection
from backend.auth import authenticate
from bson import ObjectId

# BERT
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load fine-tuned model
model_path = "./bert_model"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
model.eval()  # Evaluation mode

app = FastAPI()

# Pydantic Models
class Product(BaseModel):
    product_name: str
    category: str

class Review(BaseModel):
    product_name: str
    review_text: str

# Sentiment prediction function
def predict_sentiment(text):
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        output = model(**encoding)
    prediction = torch.argmax(output.logits, dim=-1).item()
    return "Positive" if prediction == 1 else "Negative"

# Root endpoint
@app.get("/")
def home():
    return {"message": "Welcome to the Product Review API"}

# Add product (auth required)
@app.post("/add_product/")
def add_product(product: Product, username: str = Depends(authenticate)):
    product_data = product.dict()
    products_collection.insert_one(product_data)
    return {"message": f"{product.product_name} added successfully!"}

# Delete product (auth required)
@app.delete("/delete_product/{product_name}")
def delete_product(product_name: str, username: str = Depends(authenticate)):
    result = products_collection.delete_one({"product_name": product_name})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Product not found")
    return {"message": f"{product_name} deleted successfully!"}

# Submit review (anyone)
@app.post("/submit_review/")
def submit_review(review: Review):
    sentiment = predict_sentiment(review.review_text)
    review_data = {
        "product_name": review.product_name,
        "review_text": review.review_text,
        "sentiment": sentiment
    }
    reviews_collection.insert_one(review_data)
    return {"message": "Review submitted successfully!", "sentiment": sentiment}

# Get list of product names (for dropdown in frontend)
@app.get("/get_products/")
def get_products():
    products = products_collection.find({}, {"_id": 0, "product_name": 1})
    product_names = [p["product_name"] for p in products]
    return {"products": product_names}