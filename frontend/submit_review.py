import tkinter as tk
from tkinter import ttk, messagebox
import requests

# FastAPI Endpoints
API_URL = "http://127.0.0.1:8000/submit_review/"
FETCH_PRODUCTS_URL = "http://127.0.0.1:8000/get_products/"

# Fetch products from API
def fetch_products():
    try:
        response = requests.get(FETCH_PRODUCTS_URL)
        if response.status_code == 200:
            return response.json()["products"]
    except Exception as e:
        print("Error fetching products:", e)
    return []

# Submit review to API
def submit_review():
    product = product_dropdown.get()
    review_text = review_entry.get("1.0", tk.END).strip()

    if not product or not review_text:
        messagebox.showerror("Error", "Please enter a review and select a product.")
        return

    review_data = {"product_name": product, "review_text": review_text}
    try:
        response = requests.post(API_URL, json=review_data)
        if response.status_code == 200:
            messagebox.showinfo("Success", "Review submitted successfully!")
        else:
            messagebox.showerror("Error", "Failed to submit review.")
    except Exception as e:
        messagebox.showerror("Error", f"Request failed: {e}")

# GUI Setup
root = tk.Tk()
root.title("Submit Product Review")
root.geometry("500x300")

tk.Label(root, text="Select Product:").pack()
product_dropdown = ttk.Combobox(root, values=fetch_products(), state="readonly")
product_dropdown.pack(pady=5)

tk.Label(root, text="Enter Review:").pack()
review_entry = tk.Text(root, height=5, width=50)
review_entry.pack(pady=5)

tk.Button(root, text="Submit Review", command=submit_review).pack(pady=10)

root.mainloop()
