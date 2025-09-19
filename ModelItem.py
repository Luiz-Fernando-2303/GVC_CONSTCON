from typing import Optional, List
from Property import Property
from Mesh import Mesh
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import json

executor = ThreadPoolExecutor()
with open("files/nested_codes.json", "r", encoding="utf-8") as f:
    codes = json.load(f)

class ModelItem:
    """ModelItem type"""

    meshData: bytes
    mesh: Mesh
    image: bytes
    properties: List[Property]

    def __init__(self, meshData: Optional[bytes], image: Optional[bytes], properties: List[Property]):
        self.meshData = meshData
        self.image = image
        self.properties = properties
        self.mesh = Mesh()

        if meshData is not None and len(meshData) > 0:
            self.mesh.decodeMeshData(meshData)
            
class ClassificationItem:
    """ClassificationItem type for training"""

    label: str  # Class label (value from some of the properties, represents the desired classification)
    properties: List[Property]  # List of properties (except label)
    cods: List[Property]
    vectorizedProperties: List[np.ndarray] # List of vectorized properties
    ClassificationLevels: List
    vectorsize: int

    vectorizer: TfidfVectorizer # used vectorizer for instance

    def __init__(self, label: str, properties: List[Property]):
        self.label = label
        self.properties = properties
        self.ClassificationLevels = [level for level in label.split("-") if level != ""]
        [self.cods.append(p) for p in self.createTermList()]
        self.vectorsize = sum([v.size for v in self.properties])

    def createTermList(self) -> List[Property]:
        current = codes.get("3E", {})
        result: List[Property] = []

        for level in self.ClassificationLevels:
            if level == "3E":
                continue

            if level in current:  
                current = current[level]
                
                if "desc" in current:
                    result.append(Property("COD", level, current["desc"]))
            else:
                break

        return result