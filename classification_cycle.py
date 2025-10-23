import json
import re
from typing import List, Dict
from fuzzywuzzy import fuzz
from Types import GvcObject, Property
from sentence_transformers import SentenceTransformer, util
from Db import DBManager
import os
import torch

EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

IGNORED_KEYWORDS = [
    "id", "guid", "handle", "path", "element", "category id",
    "family id", "internal", "level", "host", "parameter id", "unique id", "layer",
    "required", "icon", "true", "false", "null", "double", "integer", "string",
]

NUMERIC_PATTERN = re.compile(r"^\d+([.,]\d+)?$")

CACHE_PATH = "files/embeddings_cache.json"
 
CODES = json.load(open("files/codes.json", "r", encoding="utf-8"))

def is_useful_property(prop: Property) -> bool:
    if not prop or not prop.Info:
        return False
    if len(prop.Info.strip()) < 3:
        return False
    if NUMERIC_PATTERN.match(prop.Info.strip()):
        return False
    name = (prop.Name or "").lower()
    cat = (prop.Category or "").lower()
    if any(kw in name for kw in IGNORED_KEYWORDS):
        return False
    if any(kw in cat for kw in IGNORED_KEYWORDS):
        return False
    if any(kw in prop.Info.lower() for kw in IGNORED_KEYWORDS):
        return False
    return True

def is_useful(string: str) -> bool:
    if not string:
        return False
    if len(string.strip()) < 3:
        return False
    if NUMERIC_PATTERN.match(string.strip()):
        return False
    if any(kw in string.lower() for kw in IGNORED_KEYWORDS):
        return False
    return True

def load_embeddings_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_embeddings_cache(cache):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def GenerateResults(obj, codes, fuzzy_weight=0.4, semantic_weight=0.6, min_score=40, embed_model=None):
    texts = [p.Info for p in obj.Properties if is_useful_property(p)]
    if not texts:
        return {"codes": [], "fuzzy_scores": [], "semantic_scores": [], "descriptions": [], "ratios": []}

    combined_text = " ".join(map(str, texts)).lower().strip()
    if not combined_text:
        return {"codes": [], "fuzzy_scores": [], "semantic_scores": [], "descriptions": [], "ratios": []}

    descriptions = list(codes.values())

    fuzzy_scores_list = [fuzz.token_set_ratio(combined_text, d) for d in descriptions] if fuzzy_weight > 0 else [0] * len(descriptions)

    semantic_scores_list = [0] * len(descriptions)
    if semantic_weight > 0 and embed_model:
        cache = load_embeddings_cache()

        uncached = [d for d in descriptions if d not in cache]
        if uncached:
            new_embeddings = embed_model.encode(uncached, convert_to_tensor=False).tolist()  # batch
            for desc, emb in zip(uncached, new_embeddings):
                cache[desc] = emb

        save_embeddings_cache(cache)

        input_embedding = embed_model.encode(combined_text, convert_to_tensor=True).to('cuda')
        desc_embeddings_tensor = torch.tensor([cache[d] for d in descriptions]).to('cuda')
        semantic_scores_list = (util.cos_sim(input_embedding, desc_embeddings_tensor)[0] * 100).cpu().tolist()

    final_scores = {desc: f_score * fuzzy_weight + s_score * semantic_weight
                    for desc, f_score, s_score in zip(descriptions, fuzzy_scores_list, semantic_scores_list)}

    top_items = sorted(((desc, score) for desc, score in final_scores.items() if score >= min_score),
                       key=lambda x: x[1], reverse=True)[:5]

    desc_to_idx = {d: i for i, d in enumerate(descriptions)}

    return {
        "codes": [list(codes.keys())[desc_to_idx[desc]] for desc, _ in top_items],
        "fuzzy_scores": [fuzzy_scores_list[desc_to_idx[desc]] for desc, _ in top_items],
        "semantic_scores": [semantic_scores_list[desc_to_idx[desc]] for desc, _ in top_items],
        "descriptions": [desc for desc, _ in top_items],
        "ratios": [score for _, score in top_items]
    }

def GenerateResultsFromTexts(texts, codes, fuzzy_weight=0.4, semantic_weight=0.6, min_score=40, embed_model=None):
    if not texts:
        return {"codes": [], "fuzzy_scores": [], "semantic_scores": [], "descriptions": [], "ratios": []}

    texts = [t for t in texts if is_useful(t)]
    combined_text = " ".join(map(str, texts)).lower().strip()
    if not combined_text:
        return {"codes": [], "fuzzy_scores": [], "semantic_scores": [], "descriptions": [], "ratios": []}

    descriptions = list(codes.values())

    fuzzy_scores_list = [fuzz.token_set_ratio(combined_text, d) for d in descriptions] if fuzzy_weight > 0 else [0] * len(descriptions)

    semantic_scores_list = [0] * len(descriptions)
    if semantic_weight > 0 and embed_model:
        cache = load_embeddings_cache()

        uncached = [d for d in descriptions if d not in cache]
        if uncached:
            new_embeddings = embed_model.encode(uncached, convert_to_tensor=False).tolist()
            for desc, emb in zip(uncached, new_embeddings):
                cache[desc] = emb

        save_embeddings_cache(cache)

        input_embedding = embed_model.encode(combined_text, convert_to_tensor=True).to('cuda')
        desc_embeddings_tensor = torch.tensor([cache[d] for d in descriptions]).to('cuda')
        semantic_scores_list = (util.cos_sim(input_embedding, desc_embeddings_tensor)[0] * 100).cpu().tolist()

    final_scores = {
        desc: f_score * fuzzy_weight + s_score * semantic_weight
        for desc, f_score, s_score in zip(descriptions, fuzzy_scores_list, semantic_scores_list)
    }

    top_items = sorted(((desc, score) for desc, score in final_scores.items() if score >= min_score),
                       key=lambda x: x[1], reverse=True)[:5]

    desc_to_idx = {d: i for i, d in enumerate(descriptions)}

    return {
        "codes": [list(codes.keys())[desc_to_idx[desc]] for desc, _ in top_items],
        "fuzzy_scores": [fuzzy_scores_list[desc_to_idx[desc]] for desc, _ in top_items],
        "semantic_scores": [semantic_scores_list[desc_to_idx[desc]] for desc, _ in top_items],
        "descriptions": [desc for desc, _ in top_items],
        "ratios": [score for _, score in top_items]
    }