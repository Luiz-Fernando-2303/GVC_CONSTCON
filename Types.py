from typing import List, Optional, Tuple
import numpy as np
import base64
from transformers import BertModel
from torch.utils.data import Dataset
import torch
import torch.nn as nn

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, label_encoder, max_len=128):
        self.texts = texts
        self.labels = label_encoder.transform(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
            return_attention_mask=True
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

class TransformerClassifier(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        return self.fc(x)

class TrainingData:
    def __init__(self, propText: str, code: str):
        self.PropText = propText
        self.Code = code

    @staticmethod
    def build(items: list):
        return [TrainingData(i.get("propText", ""), i.get("expectedCode", "")) for i in items]

    def __repr__(self):
        return f"TrainingDataDto({self.PropText}, {self.Code})"

class Property:
    def __init__(self, category: str, name: str, info: str, property_id: Optional[int] = None):
        self.PropertyId: Optional[int] = property_id
        self.Category: str = category
        self.Name: str = name
        self.Info: str = info

    def __repr__(self):
        return f"Property(Category={self.Category}, Name={self.Name}, Info={self.Info})"
    
    def __todict__(self):
        return {
            "propertyid": self.PropertyId,
            "category": self.Category,
            "name": self.Name,
            "info": self.Info
        }
    
class Geometry:
    def __init__(self):
        self.vertices: List[Tuple[float, float, float]] = []  # [(x, y, z), ...]
        self.triangles: List[Tuple[int, int, int]] = []       # [(i0, i1, i2)]

    def vectorize(self) -> np.ndarray:
        vertex_array = np.array(self.vertices, dtype=np.float32).flatten()
        triangle_array = np.array(self.triangles, dtype=np.int32).flatten()
        return np.concatenate([vertex_array, triangle_array])

    def features(self) -> np.ndarray:
        if not self.vertices:
            return np.zeros(7, dtype=np.float32)  # fallback vazio
        vertices = np.array(self.vertices, dtype=np.float32)
        center = vertices.mean(axis=0)
        bbox = vertices.max(axis=0) - vertices.min(axis=0)
        triangle_count = len(self.triangles)
        return np.concatenate([center, bbox, [triangle_count]])

    def decode_mesh(self, mesh_data: bytes) -> None:
        self.vertices = []
        self.triangles = []

        try:
            obj_string = mesh_data.decode('utf-8')
        except UnicodeDecodeError:
            raise ValueError("mesh_data não pôde ser decodificado como UTF-8")

        for line in obj_string.splitlines():
            line = line.strip()
            if line.startswith('v '):
                parts = line.split()
                if len(parts) != 4:
                    continue
                _, x, y, z = parts
                self.vertices.append((
                    float(x.replace(',', '.')),
                    float(y.replace(',', '.')),
                    float(z.replace(',', '.'))
                ))
            elif line.startswith('f '):
                parts = line.split()[1:]
                indices = [int(p.split('/')[0]) - 1 for p in parts]
                if len(indices) == 3:
                    self.triangles.append(tuple(indices))
                elif len(indices) > 3:
                    for i in range(1, len(indices) - 1):
                        self.triangles.append((indices[0], indices[i], indices[i+1]))

    @classmethod
    def from_mesh_data(cls, mesh_data: bytes) -> 'Geometry':
        """
        Cria um objeto Geometry diretamente de mesh_data.
        """
        geom = cls()
        geom.decode_mesh(mesh_data)
        return geom

    def __repr__(self):
        return f"Geometry(vertices={len(self.vertices)}, triangles={len(self.triangles)}------)"


class PropObject:
    def __init__(self, object_id: int, property_id: int, prop_object_id: Optional[int] = None):
        self.PropObjectId: Optional[int] = prop_object_id
        self.ObjectId: int = object_id
        self.PropertyId: int = property_id

    def __repr__(self):
        return f"PropObject(ObjectId={self.ObjectId}, PropertyId={self.PropertyId})"


class GvcObject:
    def __init__(self, object_id: Optional[int] = None, name: str = "", type_: str = "", source_file: str = "", properties: Optional[List[Property]] = None, geometries: Optional[List[Geometry]] = None):
        self.ObjectId: Optional[int] = object_id
        self.Name: str = name
        self.Type: str = type_
        self.SourceFile: str = source_file
        self.Properties: List[Property] = properties or []
        self.Geometries: List[Geometry] = geometries or []

    @staticmethod
    def build(items_json) -> List['GvcObject']:
        objects_list: List[GvcObject] = []

        for item in items_json:
            gvc = GvcObject()
            gvc.ObjectId = item.get("objectid", 0) or item.get("ObjectId", 0)
            gvc.Name = item.get("name", "") or item.get("Name", "")
            gvc.Type = item.get("type", "") or item.get("Type", "")
            gvc.SourceFile = item.get("sourcefile", "") or item.get("SourceFile", "")

            properties_raw = item.get("properties", []) or item.get("Properties", [])
            gvc.Properties = [
                Property(
                    property_id=prop.get("propertyId", 0) or prop.get("PropertyId", 0),
                    category=prop.get("category", "") or prop.get("Category", ""),
                    name=prop.get("name", "") or prop.get("Name", ""),
                    info=prop.get("info", "") or prop.get("Info", ""),
                )
                for prop in properties_raw
            ]

            geometries_raw = item.get("geometries", []) or item.get("Geometries", [])
            gvc.Geometries = []
            for geo in geometries_raw:
                geometry = Geometry()
                mesh_data_b64 = geo.get("mesh") or geo.get("Mesh")
                if mesh_data_b64:
                    mesh_data = base64.b64decode(mesh_data_b64) if isinstance(mesh_data_b64, str) else mesh_data_b64
                    geometry.decode_mesh(mesh_data)
                geometry.GeometryId = geo.get("geometryid", 0) or geo.get("GeometryId", 0)
                geometry.ObjectId = gvc.ObjectId
                geometry.Transform = geo.get("transform") or geo.get("Transform")
                gvc.Geometries.append(geometry)

            objects_list.append(gvc)

        return objects_list

    def __repr__(self):
        return f"GvcObject(Name={self.Name}, Type={self.Type}, Properties={len(self.Properties)}, Geometries={len(self.Geometries)})"