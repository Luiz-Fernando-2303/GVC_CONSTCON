import torch
import torch.nn as nn
import torch.nn.functional as F
from ModelItem import ClassificationItem
from ClassificationTools import getClassificationItems
from tqdm import tqdm
import numpy as np
from TorchComponents import ItemCODDataset, CODSimilarityModel, buildPairs
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader
import torch

ClassificationItems: list[ClassificationItem]
ClassificationItems = getClassificationItems(
    limit=5000,
    filter={},
    labelDefinition=[
        {"Category": "NBR", "Name": "COD"},
    ]
)

max_input_size = 0
for item in ClassificationItems:
    max_input_size = max(max_input_size, sum([i.vectorsize for i in item.properties]))

model = CODSimilarityModel(max_input_size, max_input_size)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()

pairs, labels = buildPairs(ClassificationItems)
dataSet = ItemCODDataset(pairs, labels)
loader = DataLoader(dataSet, batch_size=32, shuffle=True)

for epoch in range(10):
    for item_vec, cod_vec, y in tqdm(loader, desc=f"Epoch {epoch}"):
        optimizer.zero_grad()
        y_pred = model(item_vec, cod_vec)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        