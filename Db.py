import requests
import base64
from typing import List
from tqdm import tqdm
from Types import GvcObject, Property, Geometry

API = "http://ec2-34-229-178-60.compute-1.amazonaws.com:5000"
BASE_URL_GETALL = f"{API}/modelitems/getall"
BASE_URL_GETBYFILTER = f"{API}/modelitems/filter"

class DBManager:
    @staticmethod
    def _getItems(limit: int = 0) -> List[GvcObject]:
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
                    property_id=prop.get("propertyId", 0),
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

    @staticmethod
    def _getItemsByFilter(filter: dict, limit: int = 0) -> List[GvcObject]:
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

