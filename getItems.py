import requests
import base64
from typing import List
from tqdm import tqdm
from Types import GvcObject, Property, Geometry

LOCALHOST = "http://localhost:5066"
BASE_URL_GETALL = f"{LOCALHOST}/modelitems/getall"
BASE_URL_GETBYFILTER = f"{LOCALHOST}/modelitems/filter"

def getItems(limit: int = 0) -> List[GvcObject]:
    url = BASE_URL_GETALL
    if limit > 0:
        url += f"?limit={limit}"

    resp = requests.get(url)
    print("Status:", resp.status_code)
    if not resp.ok:
        print("Erro na requisição:", resp.text)
        return []

    items = resp.json()
    objects_list: List[GvcObject] = []

    for item in tqdm(items, desc="Processando itens"):
        gvc = GvcObject()
        gvc.ObjectId = item.get("objectid", 0)
        gvc.Name = item.get("name", "")
        gvc.Type = item.get("type", "")
        gvc.SourceFile = item.get("sourcefile", "")

        properties_raw = item.get("properties", [])
        gvc.Properties = [
            Property(
                property_id=prop.get("propertyid", 0),
                category=prop.get("category", ""),
                name=prop.get("name", ""),
                info=prop.get("info", "")
            )
            for prop in properties_raw
        ]

        geometries_raw = item.get("geometries", [])
        gvc.Geometries = []
        for geo in geometries_raw:
            geometry = Geometry()
            mesh_data_b64 = geo.get("mesh")
            if mesh_data_b64:
                mesh_data = base64.b64decode(mesh_data_b64) if isinstance(mesh_data_b64, str) else mesh_data_b64
                geometry.decode_mesh(mesh_data)
            geometry.GeometryId = geo.get("geometryid", 0)
            geometry.ObjectId = gvc.ObjectId
            geometry.Transform = geo.get("transform")
            gvc.Geometries.append(geometry)

        objects_list.append(gvc)

    return objects_list


def getItemsByFilter(filter: dict, limit: int = 0) -> List[GvcObject]:
    if not filter:
        raise ValueError("Filter é obrigatório")

    url = BASE_URL_GETBYFILTER
    if limit > 0:
        url += f"?limit={limit}"

    resp = requests.post(url, json=filter)
    print("Status:", resp.status_code)
    if not resp.ok:
        print("Erro na requisição:", resp.text)
        return []

    items = resp.json()
    objects_list: List[GvcObject] = []

    for item in tqdm(items, desc="Processando itens filtrados"):
        gvc = GvcObject()
        gvc.ObjectId = item.get("objectid", 0)
        gvc.Name = item.get("name", "")
        gvc.Type = item.get("type", "")
        gvc.SourceFile = item.get("sourcefile", "")

        properties_raw = item.get("properties", [])
        gvc.Properties = [
            Property(
                property_id=prop.get("propertyid", 0),
                category=prop.get("category", ""),
                name=prop.get("name", ""),
                info=prop.get("info", "")
            )
            for prop in properties_raw
        ]

        geometries_raw = item.get("geometries", [])
        gvc.Geometries = []
        for geo in geometries_raw:
            geometry = Geometry()
            mesh_data_b64 = geo.get("mesh")
            if mesh_data_b64:
                mesh_data = base64.b64decode(mesh_data_b64) if isinstance(mesh_data_b64, str) else mesh_data_b64
                geometry.decode_mesh(mesh_data)
            geometry.GeometryId = geo.get("geometryid", 0)
            geometry.ObjectId = gvc.ObjectId
            geometry.Transform = geo.get("transform")
            gvc.Geometries.append(geometry)

        objects_list.append(gvc)

    return objects_list


if __name__ == "__main__":

    print("Test 1: Get all items (limit=5)")
    try:
        all_items = getItems(limit=5)
        print(f"Total items returned: {len(all_items)}\n")
        for i, item in enumerate(all_items, start=1):
            print(f"Item {i}:")
            print(f"  ObjectId: {item.ObjectId}")
            print(f"  Name: {item.Name}")
            print(f"  Type: {item.Type}")
            print(f"  SourceFile: {item.SourceFile}")
            print(f"  Properties: {len(item.Properties)}")
            print(f"  Geometries: {len(item.Geometries)}\n")
    except Exception as e:
        print("Error getting all items:", e)

    print("\nTest 2: Get items by filter")
    test_filter = {
        "gvcobject": {
            "name": ["TQS - Pilar retangular"]
        },
        "property": {
            "category": ["Custom"],
            "name": ["NBR_COD"],
            "info": ["3E.42.06.00.00.00.00"]
        }
    }

    try:
        filtered_items = getItemsByFilter(test_filter, limit=1000)
        print(f"Total filtered items returned: {len(filtered_items)}\n")
        for i, item in enumerate(filtered_items, start=1):
            print(f"Item {i}:")
            print(f"  ObjectId: {item.ObjectId}")
            print(f"  Name: {item.Name}")
            print(f"  Type: {item.Type}")
            print(f"  SourceFile: {item.SourceFile}")
            print(f"  Properties: {len(item.Properties)}")
            print(f"  Geometries: {len(item.Geometries)}\n")
    except Exception as e:
        print("Error getting filtered items:", e)
