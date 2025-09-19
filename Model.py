import tensorflow as tf
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense       # type: ignore
from tensorflow.keras.layers import Input       # type: ignore
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import os
import json
from ModelItem import ClassificationItem
from ClassificationTools import getClassificationItems
from tqdm import tqdm

ClassificationItems: list[ClassificationItem]
ClassificationItems = getClassificationItems(
    limit=5000,
    filter={},
    labelDefinition=[
        {"Category": "NBR", "Name": "COD"},
    ]
)

def CreateXY(ClassificationItems: list[ClassificationItem]) -> tuple[np.ndarray, np.ndarray]:
    X = []
    y = []

    max_len = max([sum([v.size for v in item.vectorizedProperties]) for item in ClassificationItems])

    for item in tqdm(ClassificationItems, desc="Creating X and y"):
        vector = np.concatenate(item.vectorizedProperties)
        if vector.size < max_len:
            vector = np.pad(vector, (0, max_len - vector.size))
        elif vector.size > max_len:
            vector = vector[:max_len]
        X.append(vector)
        y.append(item.label)

    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    return X, y

def GetEncodedLabels(y) -> tuple[np.ndarray, OneHotEncoder]:
    encoder = OneHotEncoder()
    y_encoded = encoder.fit_transform(y).toarray() 
    return y_encoded, encoder

def CreateModel(X: np.ndarray, y_encoded: np.ndarray, encoder: OneHotEncoder) -> tuple[Sequential, OneHotEncoder, np.ndarray, np.ndarray]:
    model = Sequential([
        Input(shape=(X.shape[1],)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(len(encoder.categories_[0]), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model, encoder, X, y_encoded

def TrainModel(
        model: Sequential,
        X: np.ndarray,
        y_encoded: np.ndarray,
        encoder: OneHotEncoder,
        epochs: int = 10,
        batch_size: int = 32
    ) -> Sequential:
    
    model.fit(X, y_encoded, epochs=epochs, batch_size=batch_size, verbose=1)
    return model

def classify(model: Sequential, encoder: OneHotEncoder, item: ClassificationItem, max_len: int) -> str:
    vector = np.concatenate(item.vectorizedProperties)
    if vector.size < max_len:
        vector = np.pad(vector, (0, max_len - vector.size))
    elif vector.size > max_len:
        vector = vector[:max_len]
    vector = vector.reshape(1, -1)

    pred_probs = model.predict(vector)
    pred_index = np.argmax(pred_probs, axis=1)[0]

    return encoder.categories_[0][pred_index]


def UseCompiledModel(model_path: str, data_path: str, item: ClassificationItem) -> str:
    model = tf.keras.models.load_model(model_path)
    with open(data_path, "r") as f:
        data = json.load(f)
    
    max_len = data["max_len"]
    vector = np.concatenate(item.vectorizedProperties)
    if vector.size < max_len:
        vector = np.pad(vector, (0, max_len - vector.size))
    elif vector.size > max_len:
        vector = vector[:max_len]
    vector = vector.reshape(1, -1)

    pred_probs = model.predict(vector)
    pred_index = np.argmax(pred_probs, axis=1)[0]

    return data["encoder_categories"][pred_index]


if __name__ == "__main__":
    X, y = CreateXY(ClassificationItems)
    y_encoded, encoder = GetEncodedLabels(y)
    model, encoder, X, y_encoded = CreateModel(X, y_encoded, encoder)
    model = TrainModel(model, X, y_encoded, encoder, epochs=32, batch_size=64)

    model.save(os.path.join("compiled", "model.keras"))
    data = {
        "encoder_categories": encoder.categories_[0].tolist(),
        "max_len": X.shape[1]
    }

    with open(os.path.join("compiled", "data.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
