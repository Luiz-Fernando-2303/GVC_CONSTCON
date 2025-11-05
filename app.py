from flask import Flask, jsonify, request
from Types import GvcObject
from classification_cycle import *
from predict import *
from neuralnet import *
from threading import Thread

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

@app.route("/classify", methods=["POST"]) # POST to http://ec2-54-166-164-195.compute-1.amazonaws.com:5500/classify
def classify():
    try:
        data = request.get_json(force=True)

        if isinstance(data, list) and all(isinstance(x, dict) for x in data):
            objects = GvcObject.build(data)

            predict_data : dict[str, str] = {}

            for object in objects:
                propTexts = [prop.Info for prop in object.Properties]
                propTexts = ', '.join([prop.Info for prop in object.Properties])
                predict_data[object.Name] = propTexts

            results : dict[str, str] = {}
            for data in predict_data:
                text = predict_data[data]
                result : List[str] = predict_list([text])[0]
                results[data] = result

            return jsonify(results)
        
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            result = predict_list(data)
            return jsonify(result)
        
        if isinstance(data, str):
            result = predict_list([data])
            return jsonify({data: result[0]})

        return jsonify({"error": "Formato inválido. Envie um JSON de objeto(s) ou lista de textos."}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/model/train", methods=["GET"]) # curl -X GET http://ec2-54-166-164-195.compute-1.amazonaws.com:8000/model/train
def train():
    def background_train():
        try:
            print("[INFO] Starting to fetch training data...")
            texts, labels = get_training_data(limit=100000)
            print(f"[INFO] Data fetched: {len(texts)} texts, {len(labels)} labels")

            print("[INFO] Starting model training...")
            train_model(texts, labels, epochs=8, batch_size=16)
            print("[INFO] Model training completed successfully!")
        except Exception as e:
            print("[ERROR] Training error:", e)

    print("[INFO] Training started in background")
    Thread(target=background_train).start()
    return jsonify({"status": "started"}), 200
    
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    import torch
    torch.set_num_threads(1)
    app.run(host="0.0.0.0", port=5500, debug=False)

    