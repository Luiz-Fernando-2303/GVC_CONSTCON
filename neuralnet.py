import requests
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from tqdm import tqdm
from Types import *


API = "http://ec2-34-229-178-60.compute-1.amazonaws.com:5000/modelitems" 
# API = "http://localhost:5066/modelitems"
BASE_URL_TRAININGDATA = f"{API}/trainData/get"
MODEL_DIR = "model_transformer"
os.makedirs(MODEL_DIR, exist_ok=True)

def get_training_data(code=None, limit=0):

    url = BASE_URL_TRAININGDATA

    params = {}
    if code:
        params["code"] = code
    if limit > 0:
        params["limit"] = limit

    try:
        resp = requests.get(url, params=params)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise Exception(f"Falha na API: {e}")

    data = resp.json()
    
    texts = [d["propText"] for d in data if "propText" in d and d["propText"]]
    labels = [d["code"] for d in data if "code" in d and d["code"]]

    return texts, labels

def train_model(texts, labels, epochs=2, batch_size=8):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)

    dataset = TextDataset(texts, labels, tokenizer, label_encoder)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TransformerClassifier(num_labels=len(label_encoder.classes_))
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "model.pt"))
    tokenizer.save_pretrained(MODEL_DIR)
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))
    return model, tokenizer, label_encoder

def predict(text, model, tokenizer, label_encoder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        preds = torch.argmax(outputs, dim=1)
    return label_encoder.inverse_transform(preds.cpu().numpy())[0]

if __name__ == "__main__":
    texts, labels = get_training_data(limit=100000)
    model, tokenizer, label_encoder = train_model(texts, labels, epochs=8, batch_size=32)

    exemplo = "Paredes estruturais de concreto moldado in loco"
    pred = predict(exemplo, model, tokenizer, label_encoder)
    print(f"Predição para '{exemplo}': {pred}")