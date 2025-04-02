import kagglehub
import shutil
import os

custom_path = "data"

downloaded_path = kagglehub.dataset_download("bittlingmayer/amazonreviews")

if not os.path.exists(custom_path):
    os.makedirs(custom_path)  

shutil.move(downloaded_path, custom_path)

print(f"Dataset saved in: {custom_path}")
