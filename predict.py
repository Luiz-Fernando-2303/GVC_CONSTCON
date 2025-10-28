import torch
from transformers import BertTokenizer
import joblib
import os
from Types import *  

MODEL_DIR = "model_transformer"

def load_model_components():
    from transformers import BertModel
    import torch.nn as nn

    class TransformerClassifier(nn.Module):
        def __init__(self, num_labels):
            super(TransformerClassifier, self).__init__()
            self.bert = BertModel.from_pretrained("bert-base-uncased")
            self.dropout = nn.Dropout(0.3)
            self.fc = nn.Linear(self.bert.config.hidden_size, num_labels)

        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output
            out = self.dropout(pooled_output)
            return self.fc(out)

    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

    num_labels = len(label_encoder.classes_)
    model = TransformerClassifier(num_labels)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "model.pt"), map_location="cpu"))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, tokenizer, label_encoder, device


def predict_list(texts : list[str]) -> list[str]:
    model, tokenizer, label_encoder, device = load_model_components()

    encodings = tokenizer(
        texts,
        add_special_tokens=True,
        max_length=128,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )

    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        preds = torch.argmax(outputs, dim=1)

    predicted_labels = label_encoder.inverse_transform(preds.cpu().numpy())
    return predicted_labels.tolist()

if __name__ == "__main__":
    exemplos = [
        "Paredes estruturais de concreto moldado in loco",
        "Viga de aço galvanizado",
        "Piso cerâmico esmaltado"
    ]
    preds = predict_list(exemplos)
    for texto, label in zip(exemplos, preds):
        print(f"{texto} → {label}")
