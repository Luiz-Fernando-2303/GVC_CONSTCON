import requests
import json

BASE_URL = "http://127.0.0.1:5000"

def pretty(data):
    print(json.dumps(data, indent=2, ensure_ascii=False))

def test_health():
    print("Testing /health ...")
    resp = requests.get(f"{BASE_URL}/health")
    pretty(resp.json())

def test_metadata():
    print("\nTesting /gvcobject/metadata ...")
    resp = requests.get(f"{BASE_URL}/gvcobject/metadata")
    data = resp.json()
    print(f"Categorias: {len(data.get('categories', []))} | Names: {len(data.get('names', []))}")
    pretty({k: v[:10] if isinstance(v, list) else v for k, v in data.items()})

def test_codes(limit=10):
    print(f"\nTesting /gvcobject/codes?limit={limit} ...")
    resp = requests.get(f"{BASE_URL}/gvcobject/codes", params={"limit": limit})
    data = resp.json()
    codes = data.get("codes", [])
    print(f"Total returned: {len(codes)}")
    if codes:
        print("First codes:", codes[:3])
    return codes

def test_classify(code: str):
    print(f"\nTesting /gvcobject/classify with code='{code}' ...")
    resp = requests.post(f"{BASE_URL}/gvcobject/classify", json={"code": code})
    if resp.status_code != 200:
        print(f"Error ({resp.status_code}): {resp.text}")
        return
    data = resp.json()
    print(f"Code: {data['code']} | Props: {len(data['props'])}")
    pretty(data["props"][:5])

def test_similarity(texts):
    print(f"\nTesting /gvcobject/similarity with texts={texts} ...")
    payload = {
        "infos": texts,
        "top": 5
    }
    resp = requests.post(f"{BASE_URL}/gvcobject/similarity", json=payload)
    if resp.status_code != 200:
        print(f"Error ({resp.status_code}): {resp.text}")
        return
    data = resp.json()
    print("Top results:")
    pretty(data["top"])

def test_reset():
    print("\nTesting /gvcobject/reset ...")
    resp = requests.post(f"{BASE_URL}/gvcobject/reset")
    pretty(resp.json())


if __name__ == "__main__":
    print("Initializing tests against API Flask...\n")

    test_health()
    test_metadata()

    codes = test_codes(limit=5)
    if codes:
        test_classify(codes[0])

    test_similarity(["concrete pair", "pre-molded slab", "metal pillar"])
