import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
from ModelItem import ClassificationItem

class ItemCODDataset(Dataset):
    def __init__(self, pairs, labels):
        self.pairs = pairs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item_vec, cod_vec = self.pairs[idx]
        y = self.labels[idx]
        return (
            torch.tensor(item_vec, dtype=torch.float32),
            torch.tensor(cod_vec, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )

class CODSimilarityModel(nn.Module):
    def __init__(self, item_vec_size, cod_vec_size, hidden_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(item_vec_size + cod_vec_size + item_vec_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, item_vec, cod_vec):
        diff = torch.abs(item_vec - F.pad(cod_vec, (0, item_vec.size(-1) - cod_vec.size(-1))))
        x = torch.cat([item_vec, cod_vec, diff], dim=-1)
        return self.fc(x)
    
def buildPairs(items: List[ClassificationItem], max_input_size: int) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[int]]:
    pairs = []
    labels = []

    for item in tqdm(items, desc="Building pairs and labels"):
        item_vec = np.concatenate([v for v in item.vectorized_properties])
        item_vec = resizeArray(item_vec, max_input_size)

        for cod in item.cods:
            cod_vec = cod.vectorize(item.vectorizer)
            cod_vec = resizeArray(cod_vec, max_input_size)
            pairs.append((item_vec, cod_vec))
            labels.append(1)

        for other_item in items:
            if other_item == item:
                continue

            for cod in other_item.cods:
                cod_vec = cod.vectorize(item.vectorizer)
                cod_vec = resizeArray(cod_vec, max_input_size)
                pairs.append((item_vec, cod_vec))
                labels.append(0)

    return pairs, labels

def resizeArray(ar: np.ndarray, size: int) -> np.ndarray:
    if ar.size < size:
        return np.pad(ar, (0, size - ar.size))
    elif ar.size > size:
        return ar[:size]
    else:
        return ar       