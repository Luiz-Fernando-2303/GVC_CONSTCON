from flask import Flask, jsonify, request
from Types import GvcObject
from classification_cycle import *

app = Flask(__name__)


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json(force=True)

        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            result = GenerateResultsFromTexts(
                texts=data,
                codes=CODES,
                fuzzy_weight=0.4,
                semantic_weight=0.6,
                min_score=10,
                embed_model=EMBED_MODEL
            )
            return jsonify(result)

        if isinstance(data, dict):
            data = [data]

        if isinstance(data, list) and all(isinstance(x, dict) for x in data):
            objects = GvcObject.build(data)

            results = [
                GenerateResults(
                    obj=obj,
                    codes=CODES,
                    fuzzy_weight=0.4,
                    semantic_weight=0.6,
                    min_score=10,
                    embed_model=EMBED_MODEL
                )
                for obj in objects
            ]

            return jsonify(results[0] if len(results) == 1 else results)

        return jsonify({"error": "Formato inválido. Envie um JSON de objeto(s) ou lista de textos."}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    torch.set_num_threads(1)
    app.run(host="0.0.0.0", port=5000, debug=True)