from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from uuid import uuid4
from query_data import query_rag
import os
import sys

app = Flask(__name__, static_folder="../frontend/build", static_url_path="/")
CORS(app)


@app.route("/")
def api_index():
    return app.send_static_file("index.html")


@app.route("/api/chat", methods=["POST"])
def api_chat():
    request_json = request.get_json()
    query_text = request_json.get("question")
    if query_text is None:
        return jsonify({"msg": "Missing question from request JSON"}), 400

    # session_id = request.args.get("session_id", str(uuid4()))
    return Response(query_rag(query_text), mimetype="text/event-stream")


@app.cli.command()
def create_index():
    """Create or re-create the index."""
    basedir = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(f"{basedir}/../")

    from data import index_data

    index_data.main()


if __name__ == "__main__":
    app.run(port=3001, debug=True)
