from Db import DBManager
from Types import GvcObject, Property, Geometry

import re
import unicodedata
from collections import Counter
from typing import Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def is_descriptive(text: str) -> bool:
    if not text or not isinstance(text, str):
        return False

    text = text.strip().lower()
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')

    allowed_strings = ["est", "ele", "hid", "arq"]
    denied_strings = ["nulo", "true", "false", "null", "none", "nao", "nenhuma", "sem"]

    if len(text) < 3:
        return False

    try:
        _ = float(text.replace(",", "."))
        return False
    except ValueError:
        pass

    if any(word in denied_strings for word in text.split()):
        return False

    if any(word in allowed_strings for word in text.split()):
        return True

    if re.fullmatch(r"[a-f0-9]{8,}", text.replace("-", "").replace("_", "")):
        return False

    symbol_count = sum(c in "-_/:." for c in text)
    if symbol_count > len(text) * 0.4:
        return False

    if not re.search(r"[a-z]", text):
        return False

    words = text.split()
    if not any(len(w) > 2 for w in words):
        return False

    return True

def normalize_info(text: str) -> str:
    if not text:
        return ""

    text = text.lower()
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    text = re.sub(r'[_\-\|\\\/;,\[\]\(\)]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def GenerateExtraSamples(limit: int = 10000, items: Optional[List[GvcObject]] = None, levels_to_keep: int = 1) -> List[GvcObject]:
    items: List[GvcObject] = items or DBManager._getItems(limit)
    if not items:
        return []

    extraSamples: List[GvcObject] = []
    for item in items:
        code_props = [prop for prop in item.Properties if prop.Name == "NBR_COD"]
        if not code_props:
            continue

        clone = GvcObject()
        clone.Name = item.Name
        clone.Type = item.Type
        clone.SourceFile = item.SourceFile
        clone.Properties = [prop for prop in item.Properties if prop not in code_props]
        clone.Geometries = item.Geometries

        newProps: List[Property] = []
        for prop in code_props:
            name = prop.Name
            category = prop.Category

            code_levels = prop.Info.split('-')[1:]
            total_levels = 6
            code_levels += ['00'] * (total_levels - len(code_levels))
            new_code_levels = code_levels[:levels_to_keep] + ['00'] * (total_levels - levels_to_keep)
            info = '3E-' + '-'.join(new_code_levels)

            newProp = Property(category=category, name=name, info=info, property_id=prop.PropertyId)
            newProps.append(newProp)

        clone.Properties += newProps
        extraSamples.append(clone)

    return extraSamples

def ComumPropsPerCode(limit: int = 10000, items: Optional[List[GvcObject]] = None) -> Dict[str, List[dict[str, str]]]:
    items: List[GvcObject] = items or DBManager._getItems(limit)
    if not items:
        return {}

    codes: Dict[str, List[dict[str, str]]] = {}

    def process_item(item: GvcObject) -> tuple[str, List[dict[str, str]]]:
        code_prop = [prop for prop in item.Properties if prop.Name == "NBR_COD"]
        code = code_prop[0].Info if code_prop else "Sem Código"

        props = [
            prop.__todict__()
            for prop in item.Properties
            if prop.Name != "NBR_COD" and is_descriptive(prop.Info)
        ]
        return code, props

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_item, item) for item in items]
        for future in tqdm(as_completed(futures), total=len(items), desc="Processando itens"):
            code, props = future.result()
            if code not in codes:
                codes[code] = []
            codes[code].extend(props)

    for code, props in codes.items():
        prop_tuples = [tuple(sorted(p.items())) for p in props]
        codes[code] = [dict(t) for t, _ in Counter(prop_tuples).most_common(10000)]

    return codes

def EmbeddingMatriz(props: Dict[str, List[dict[str, str]]], model: SentenceTransformer) -> Dict[str, np.ndarray]:
    embeddings_per_code: Dict[str, np.ndarray] = {}
    for code, props_list in props.items():
        infos = [normalize_info(p["info"]) for p in props_list]
        if not infos:
            embeddings_per_code[code] = np.zeros((0, model.get_sentence_embedding_dimension()))
        else:
            embs = model.encode(infos, convert_to_numpy=True, show_progress_bar=False)
            embeddings_per_code[code] = embs

    return embeddings_per_code

def SimilarityDict(test_embeddings: np.ndarray, ref_embeddings: Dict[str, np.ndarray], skip_same: bool = False) -> Dict[str, float]:
    if test_embeddings is None or test_embeddings.size == 0:
        return {}

    if test_embeddings.ndim == 1:
        test_embeddings = test_embeddings.reshape(1, -1)

    test_mean = np.mean(test_embeddings, axis=0, keepdims=True)  # shape (1, dim)

    results: Dict[str, float] = {}
    for ref_code, emb_arr in ref_embeddings.items():
        if skip_same:
            pass

        if emb_arr is None or emb_arr.size == 0:
            results[ref_code] = 0.0
        else:
            if emb_arr.ndim == 1:
                emb_arr = emb_arr.reshape(1, -1)
            ref_mean = np.mean(emb_arr, axis=0, keepdims=True)
            sim = float(cosine_similarity(test_mean, ref_mean)[0, 0])
            results[ref_code] = sim

    return results
