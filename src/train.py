import torch
import json
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import DataLoader, TensorDataset
from model import SimpleNN
from preprocessing import load_data

X_train, X_test, y_train, y_test = load_data("data/reviews.csv")

vectorizer = CountVectorizer(max_features=2000)
X_train_vec = vectorizer.fit_transform(X_train).toarray()
X_test_vec = vectorizer.transform(X_test).toarray()

clean_vocab = {k: int(v) for k, v in vectorizer.vocabulary_.items()}

with open("models/vocab.json", "w") as f:
    json.dump(clean_vocab, f)

X_train_tensor = torch.tensor(X_train_vec, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)

dataset = TensorDataset(X_train_tensor, y_train_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = SimpleNN(input_size=X_train_vec.shape[1])

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(20):
    total_loss = 0
    for xb, yb in loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"epoch {epoch+1}, loss={total_loss:.4f}")

torch.save(model.state_dict(), "models/model.pt")

print("Моделька сохранена")