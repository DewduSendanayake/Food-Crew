from flask import Flask, request, jsonify
from main import run_crew  # assuming your function is in main.py

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to the Banana Leaf API!"})

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()

    # Validate input
    if not data or "query" not in data or "vendor" not in data:
        return jsonify({"error": "Missing 'query' or 'vendor' in request."}), 400

    query = data["query"]
    vendor = data["vendor"]

    try:
        result = run_crew(query, vendor)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
