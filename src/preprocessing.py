import re
import pandas as pd
from sklearn.model_selection import train_test_split

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zа-я0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_data(path):
    df = pd.read_csv(path)

    df["text"] = df["review_text"].apply(clean_text)

    df["label"] = df["stars"] - 1

    return train_test_split(df["text"], df["label"], test_size=0.2)