from typing import List, Dict, Any
from flask import Flask, jsonify, request
import numpy as np
import logging
from sentence_transformers import SentenceTransformer

from analysis import (
    EmbeddingMatriz,
    SimilarityDict,
    normalize_info,
    is_descriptive
)
from Db import DBManager
from Types import GvcObject, Property

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

logging.info("Carregando modelo e dados iniciais...")
_model_name = "all-mpnet-base-v2"
_model = SentenceTransformer(_model_name)

items = DBManager._getItems(limit=30000)
codes_dict = {i["code"]: i["props"] for i in items if "code" in i and "props" in i}

_embeddings_cache = EmbeddingMatriz(codes_dict, _model)

@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Recebe:
    {
        "infos": ["texto1", "texto2", ...],
        "top": 10
    }
    Retorna:
    [
        { "code": "<codigo>", "similarity": 0.95 },
        ...
    ]
    """
    body = request.get_json(force=True, silent=True)
    if not body:
        return jsonify({"error": "JSON body esperado"}), 400

    infos = body.get("infos")
    if not infos or not isinstance(infos, list):
        return jsonify({"error": "campo 'infos' deve ser uma lista de strings"}), 400

    top_n = int(body.get("top", 10))

    # Normaliza e filtra textos válidos
    infos_norm = [normalize_info(i) for i in infos if i and i.strip() and is_descriptive(i)]
    if not infos_norm:
        return jsonify({"error": "nenhum texto válido em 'infos'"}), 400

    # Calcula embeddings e similaridade
    test_embs = _model.encode(infos_norm, convert_to_numpy=True, show_progress_bar=False)
    sims = SimilarityDict(test_embs, _embeddings_cache)

    sorted_items = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)
    top_items = sorted_items[:top_n]

    result = [{"code": code, "similarity": float(sim)} for code, sim in top_items]
    return jsonify({"count": len(infos_norm), "top": result})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
