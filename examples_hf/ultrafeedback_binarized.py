from datasets import load_dataset
import os

cache_dir = "../data/ultrafeedback_binarize"

try:
    dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", cache_dir=cache_dir)
    print(f"The dataset is downloaded to {cache_dir}\n")
    print(f"The structure of dataset is {dataset}\n")

except Exception as e:
    print(f"Error happened when downloading: {e}")
