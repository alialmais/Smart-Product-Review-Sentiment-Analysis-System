from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")

# Create a database
db = client["product_reviews_db"]

# Create collections
db.create_collection("products")
db.create_collection("reviews")

print("Database and collections created successfully!")
