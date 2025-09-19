from typing import List, Tuple
import time
import numpy as np

class Mesh:
    def __init__(self):
        self.vertices: List[Tuple[float, float, float]] = []  # [(x, y, z), ...]
        self.triangles: List[Tuple[int, int, int]] = []       # [(i0, i1, i2)]

    def decodeMeshData(self, meshData: bytes) -> None:
        start = time.time()

        obj_string = meshData.decode('utf-8')
        self.vertices = []
        self.triangles = []

        for line in obj_string.splitlines():
            if line.startswith('v '):
                _, x, y, z = line.strip().split()
                self.vertices.append((float(x.replace(',', '.')),
                                    float(y.replace(',', '.')),
                                    float(z.replace(',', '.'))))
            elif line.startswith('f '):
                _, *indices = line.strip().split()
                # Subtrai 1 porque o formato OBJ é 1-based
                self.triangles.append(tuple(int(i) - 1 for i in indices))

        # print(f"Decoded mesh with {len(self.vertices)} vertices and {len(self.triangles)} triangles in {time.time() - start:.4f} seconds.")

    def vectorize(self) -> np.ndarray:
        vertex_array = np.array(self.vertices, dtype=np.float32).flatten()
        triangle_array = np.array(self.triangles, dtype=np.int32).flatten()
        return np.concatenate([vertex_array, triangle_array])
    
    def features(self) -> np.ndarray:
        vertices = np.array(self.vertices)
        center = vertices.mean(axis=0)
        bbox = vertices.max(axis=0) - vertices.min(axis=0)
        triangle_count = len(self.triangles)
        return np.concatenate([center, bbox, [triangle_count]])