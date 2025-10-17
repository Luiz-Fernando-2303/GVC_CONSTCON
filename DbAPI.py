from typing import List, Dict, Any, Optional
from flask import Flask, jsonify, request
import threading
import logging
import numpy as np

from Db import DBManager
from Types import GvcObject, Property, Geometry
from analysis import (
    GenerateExtraSamples,
    ComumPropsPerCode,
    EmbeddingMatriz,
    SimilarityDict,
    normalize_info,
    is_descriptive
)

from sentence_transformers import SentenceTransformer

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

class SingletonClassification:
    """
    Mantém em memória a lista de items, o dicionário de props por código e
    metadados (categories, names). Recarrega sob demanda.
    """
    _lock = threading.Lock()
    _instance = None

    def __init__(self, limit: int = 30000, levels_to_keep: int = 1):
        logging.info("Inicializando Classification cache...")
        items = DBManager._getItems(limit)
        # items = GenerateExtraSamples(limit=limit, items=items, levels_to_keep=levels_to_keep)
        self.items = items
        self.comuns: Dict[str, List[dict]] = ComumPropsPerCode(limit=limit, items=items)
        self.categories, self.names = self._get_all_categories_and_names()

        self._embeddings_cache: Optional[Dict[str, np.ndarray]] = None
        self._model_name = "all-mpnet-base-v2"
        self._model: Optional[SentenceTransformer] = None

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = SingletonClassification()
            return cls._instance

    def _get_all_categories_and_names(self):
        pairs = []
        for code, items in self.comuns.items():
            for prop in items:
                category = prop.get("category")
                name = prop.get("name")
                if not category or not name:
                    continue

                pair = {"category": category, "name": name}
                if pair not in pairs:
                    pairs.append(pair)

        categories = [p["category"] for p in pairs]
        names = [p["name"] for p in pairs]
        return categories, names

    def ensure_model(self):
        if self._model is None:
            logging.info("Carregando modelo de embeddings '%s'...", self._model_name)
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def build_embeddings_cache(self):
        """
        Cria e salva embeddings por código usando EmbeddingMatriz.
        """
        if self._embeddings_cache is not None:
            return self._embeddings_cache

        model = self.ensure_model()
        self._embeddings_cache = EmbeddingMatriz(self.comuns, model)
        return self._embeddings_cache

    def reset_cache(self):
        logging.info("Resetando classification e embeddings cache.")
        SingletonClassification._instance = None

@app.route("/gvcobject/classify", methods=["POST"])
def classify():
    """
    body JSON: { "code": "<codigo>" }
    Retorna: lista de propriedades comuns ao código solicitado.
    """
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "JSON body esperado"}), 400

    code = data.get("code")
    if not code:
        return jsonify({"error": "campo 'code' é obrigatório"}), 400

    classification = SingletonClassification.get_instance()
    if code not in classification.comuns:
        # tente normalizar se usuário passou sem prefixo "3E-..." por exemplo
        alt = code
        if not code.startswith("3E-"):
            alt = f"3E-{code}"
        if alt in classification.comuns:
            code = alt

    if code not in classification.comuns:
        return jsonify({"error": f"Código '{code}' não encontrado", "available_codes_count": len(classification.comuns)}), 404

    return jsonify({"code": code, "props": classification.comuns[code]})


@app.route("/gvcobject/metadata", methods=["GET"])
def metadata():
    """
    Retorna metadados: lista de categorias, names e contagem de códigos.
    """
    classification = SingletonClassification.get_instance()
    return jsonify({
        "categories": classification.categories,
        "names": classification.names,
        "codes_count": len(classification.comuns),
    })


@app.route("/gvcobject/codes", methods=["GET"])
def list_codes():
    """
    Lista códigos disponíveis (opcional ?limit=n)
    """
    classification = SingletonClassification.get_instance()
    limit = request.args.get("limit", type=int)
    codes = list(classification.comuns.keys())
    if limit is not None and limit > 0:
        codes = codes[:limit]
    return jsonify({"codes": codes, "count": len(codes)})


@app.route("/gvcobject/similarity", methods=["POST"])
def similarity():
    """
    body JSON:
    {
        "infos": ["texto1", "texto2", ...],  # ou um único string
        "top": 10,                           # opcional, qtd de top códigos retornados
        "skip_same": false                   # opcional
    }

    Retorna ranking de códigos por similaridade média entre os embeddings dos 'infos' e cada código.
    """
    body = request.get_json(force=True, silent=True)
    if not body:
        return jsonify({"error": "JSON body esperado"}), 400

    infos = body.get("infos")
    if infos is None:
        return jsonify({"error": "campo 'infos' é obrigatório (string ou lista de strings)"}), 400

    if isinstance(infos, str):
        infos = [infos]
    if not isinstance(infos, list) or not all(isinstance(i, str) for i in infos):
        return jsonify({"error": "campo 'infos' deve ser string ou lista de strings"}), 400

    top_n = int(body.get("top", 10))
    skip_same = bool(body.get("skip_same", False))

    classification = SingletonClassification.get_instance()

    embeddings_cache = classification.build_embeddings_cache()
    model = classification.ensure_model()

    infos_norm = [normalize_info(i) for i in infos if i and i.strip() and is_descriptive(i)]
    if not infos_norm:
        return jsonify({"error": "nenhum texto válido em 'infos'"}), 400

    test_embs = model.encode(infos_norm, convert_to_numpy=True, show_progress_bar=False)
    sims = SimilarityDict(test_embs, embeddings_cache, skip_same=skip_same)

    sorted_items = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)
    top_items = sorted_items[:top_n]

    result = [{"code": code, "similarity": float(sim)} for code, sim in top_items]
    return jsonify({"query_count": len(infos_norm), "top": result})


@app.route("/gvcobject/reset", methods=["POST"])
def reset():
    """
    Força reset do cache e reload (útil durante desenvolvimento).
    """
    SingletonClassification.get_instance().reset_cache()
    return jsonify({"status": "ok", "message": "cache resetado. A próxima chamada irá reconstruir a classificação."})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
