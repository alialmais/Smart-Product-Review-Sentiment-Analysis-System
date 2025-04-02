import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pymongo import MongoClient

# MongoDB connection
client = MongoClient("mongodb://localhost:27017")
db = client["product_reviews_db"]
reviews_collection = db["reviews"]
products_collection = db["products"]

# Streamlit page config
st.set_page_config(page_title="Product Sentiment Dashboard", layout="wide")
st.title("Product Sentiment Dashboard")

# Load reviews from MongoDB
def load_reviews():
    return list(reviews_collection.find({}, {"_id": 0}))

data = load_reviews()

if not data:
    st.warning("No reviews found in the database.")
else:
    df = pd.DataFrame(data)
    
    if "product_name" in df.columns and "sentiment" in df.columns:
        sentiment_counts = df.groupby(["product_name", "sentiment"]).size().unstack(fill_value=0)

        # Plot stacked bar chart with custom colors
        st.subheader("Sentiment Distribution by Product")
        fig, ax = plt.subplots(figsize=(8, 4))
        sentiment_counts.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            color={"Negative": "red", "Positive": "blue"}
        )
        ax.set_xlabel("Product")
        ax.set_ylabel("Number of Reviews")
        ax.set_title("Sentiment Analysis by Product")
        ax.legend(title="Sentiment", loc="upper right")  
        st.pyplot(fig)
    else:
        st.error("Required fields (product_name and sentiment) not found in the review data.")
