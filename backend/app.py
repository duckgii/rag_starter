"""Context Management RAG starter — agentic (ReAct) chat backend.

Instead of one-shot top-K chunk retrieval, /api/chat drives a ReAct agent
(see agent.py) that navigates the regulation by its structure: it searches the
table of contents, reads whole sections, and follows the cross-references each
section cites — then answers with citations to the sections it actually read.
"""
import json
import sys
from pathlib import Path

# Make the parent directory importable so we can use indexer.py / agent.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from anthropic import Anthropic
from dotenv import load_dotenv
from flask import Flask, Response, request, stream_with_context
from flask_cors import CORS

from agent import run as run_agent

load_dotenv()  # ANTHROPIC_API_KEY from .env

app = Flask(__name__)
CORS(app)
client = Anthropic()


@app.route("/api/chat", methods=["POST"])
def chat():
    user_message = request.json["message"]

    # The agent emits a stream of events: `status` (which tool it's running),
    # `delta` (answer text), and a final `done` (citations). We forward each as
    # a Server-Sent Event; the frontend renders status as progress, deltas as
    # the streaming answer, and done to attach the Sources list.
    def generate():
        for event in run_agent(user_message, client):
            yield f"data: {json.dumps(event)}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(port=5000, debug=True)
