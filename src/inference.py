import torch
import json
import re
from src.model import SimpleNN

def clean(text):
    text = text.lower()
    text = re.sub(r"[^a-zа-я0-9\s]", " ", text)
    return text

with open("models/vocab.json") as f:
    vocab = json.load(f)

def text_to_vector(text):
    vec = torch.zeros(len(vocab))
    for word in clean(text).split():
        if word in vocab:
            vec[vocab[word]] += 1
    return vec

model = SimpleNN(len(vocab))
model.load_state_dict(torch.load("models/model.pt"))
model.eval()

def predict(text):
    x = text_to_vector(text).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    return pred + 1, probs.tolist()