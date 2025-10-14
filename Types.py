from typing import List, Optional, Tuple
import numpy as np

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

    def __repr__(self):
        return f"GvcObject(Name={self.Name}, Type={self.Type}, Properties={len(self.Properties)}, Geometries={len(self.Geometries)})"