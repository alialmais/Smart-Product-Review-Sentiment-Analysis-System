import pandas as pd
import os
import bz2

def decompress_bz2(file_path, output_path):
    """ Decompresses a .bz2 file and saves it as a .txt file """
    with bz2.BZ2File(file_path, "rb") as file:
        data = file.read()
        with open(output_path, "wb") as output:
            output.write(data)
    print(f" Decompressed: {output_path}")
import pandas as pd
import os
import bz2

def decompress_bz2(file_path, output_path):
    """ Decompresses a .bz2 file and saves it as a .txt file """
    with bz2.BZ2File(file_path, "rb") as file:
        data = file.read()
        with open(output_path, "wb") as output:
            output.write(data)
    print(f" Decompressed: {output_path}")

def convert_txt_to_csv(input_txt, output_csv):
    """ Converts txt file to CSV format """
    data = []
    
    with open(input_txt, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.split(" ", 1)  
            if len(parts) == 2:
                label = parts[0].replace("__label__1", "Negative").replace("__label__2", "Positive")
                review = parts[1].strip()
                data.append([label, review])
    
    df = pd.DataFrame(data, columns=["label", "review"])
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f" {output_csv} saved successfully!")

# Define paths inside the project
data_folder = "data/7"  
train_bz2 = os.path.join(data_folder, "train.ft.txt.bz2")
test_bz2 = os.path.join(data_folder, "test.ft.txt.bz2")
train_txt = os.path.join(data_folder, "train.ft.txt")
test_txt = os.path.join(data_folder, "test.ft.txt")
train_csv = os.path.join("data", "train.csv")
test_csv = os.path.join("data", "test.csv")

decompress_bz2(train_bz2, train_txt)
decompress_bz2(test_bz2, test_txt)

convert_txt_to_csv(train_txt, train_csv)
convert_txt_to_csv(test_txt, test_csv)

print(" All files successfully converted to CSV!")
