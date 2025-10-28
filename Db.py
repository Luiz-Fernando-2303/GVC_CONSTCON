import requests
import base64
from typing import List
from tqdm import tqdm
from Types import GvcObject, Property, Geometry, TrainingData

API = "http://ec2-34-229-178-60.compute-1.amazonaws.com:5000"
BASE_URL_GETALL = f"{API}/modelitems/getall"
BASE_URL_GETBYFILTER = f"{API}/modelitems/filter"
BASE_URL_TRAININGDATA = f"{API}/trainData/get"

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
        return GvcObject.build(items)
    
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
        return GvcObject.build(items)

    @staticmethod
    def _getTrainingData(code: str = None, limit: int = 0) -> List[TrainingData]:
        url = BASE_URL_TRAININGDATA
        params = {}
        if code:
            params["code"] = code
        if limit > 0:
            params["limit"] = limit

        resp = requests.post(url, params=params)
        print("Status:", resp.status_code)

        if not resp.ok:
            print("Erro na requisição:", resp.text)
            return []

        data = resp.json()
        return TrainingData.build(data)