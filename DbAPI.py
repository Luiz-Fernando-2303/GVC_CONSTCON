from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route("/", methods=["GET"]) # how to use: http://127.0.0.1:5000
def clasify():
    return "Hello World"

if __name__ == "__main__":
    app.run(debug=True)
