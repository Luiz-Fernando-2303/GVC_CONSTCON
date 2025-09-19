import requests
import base64
from ModelItem import ModelItem
from Property import Property
from typing import List
from tqdm import tqdm

LOCALHOST = "http://localhost:5066"
BASE_URL_GETALL =       f"{LOCALHOST}/modelitems/getall"
BASE_URL_GETBYFILTER =  f"{LOCALHOST}/modelitems/filter"

def getItems(limit: int = 0) -> List[ModelItem]:
    """
    Get all model items from server.
    """
    url = BASE_URL_GETALL
    if limit and limit > 0:
        url += f"?limit={limit}"
    resp = requests.get(url)
    print("Status:", resp.status_code)

    if not resp.ok:
        print("Erro na requisição:", resp.text)
        return []

    items = resp.json()
    items_list = []

    for item in tqdm(items, desc="Processing items"):
        mesh_data_b64 = item.get("meshData")
        mesh_data = base64.b64decode(mesh_data_b64) if isinstance(mesh_data_b64, str) else mesh_data_b64

        image_data_raw = item.get("image", {}).get("data")
        image_data = base64.b64decode(image_data_raw) if isinstance(image_data_raw, str) else image_data_raw

        properties_raw = item.get("properties", [])
        properties_list: List[Property] = []

        for prop in properties_raw:
            category = prop.get("category")
            name = prop.get("name")
            value = prop.get("value")
            properties_list.append(Property(category, name, value))

        items_list.append(ModelItem(mesh_data, image_data, properties_list))

    return items_list

def getItemsByFilter(filter: dict[str, str], limit: int = 0) -> List[ModelItem]:
    """
    Get model items from server filtered by a property dictionary.
    Example filter: {"Category": "Material", "Name": "Color", "Value": "Red"}
    """
    if not filter:
        raise Exception("Filter is required")

    url = BASE_URL_GETBYFILTER
    if limit and limit > 0:
        url += f"?limit={limit}"

    resp = requests.post(url, json=filter)
    print("Status:", resp.status_code)

    if not resp.ok:
        print("Erro na requisição:", resp.text)
        return []

    items = resp.json()
    items_list = []

    for item in tqdm(items, desc="Processing items"):
        mesh_data_b64 = item.get("MeshData")
        mesh_data = base64.b64decode(mesh_data_b64) if isinstance(mesh_data_b64, str) else mesh_data_b64

        image_data_raw = item.get("Image", {}).get("Data")
        image_data = base64.b64decode(image_data_raw) if isinstance(image_data_raw, str) else image_data_raw

        properties_raw = item.get("Properties", [])
        properties_list: List[Property] = []

        for prop in properties_raw:
            category = prop.get("Category")
            name = prop.get("Name")
            value = prop.get("Value")
            properties_list.append(Property(category, name, value))

        items_list.append(ModelItem(mesh_data, image_data, properties_list))

    return items_list