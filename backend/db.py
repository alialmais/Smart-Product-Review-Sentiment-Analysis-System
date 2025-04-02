from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")

# Select Database
db = client["product_reviews_db"]

# Define Collections
products_collection = db["products"]
reviews_collection = db["reviews"]
