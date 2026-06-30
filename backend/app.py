"""Context Management RAG starter — extended chat backend.

This is the Foundations chat backend with stubs for retrieval-augmented generation.

TODO:
  1. Update SYSTEM_PROMPT with citation rules.
  2. In /api/chat: retrieve top-K chunks for the user's question.
  3. Format chunks as a numbered context block.
  4. Build the user_content with CONTEXT + QUESTION.
  5. Parse citation numbers from the answer; return them to the frontend.
"""
import json
import re
import sys
from pathlib import Path

# Make the parent directory importable so we can use indexer.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from anthropic import Anthropic
from dotenv import load_dotenv
from flask import Flask, Response, request, stream_with_context
from flask_cors import CORS

from indexer import load_index, search

load_dotenv()  # ANTHROPIC_API_KEY from .env

app = Flask(__name__)
CORS(app)
client = Anthropic()

# Load the index once at startup. Fails fast if no index — run `python indexer.py` first.
INDEX = load_index()
print(f"Loaded {len(INDEX)} chunks from disk")


# ════════════════════════════════════════════════════════════════
# TODO — update SYSTEM_PROMPT with citation rules.
# Suggestions:
#   - Answer ONLY from the provided context.
#   - Cite each factual claim with [n] using the numbers in the context.
#   - If the context doesn't contain the answer, say so explicitly.
# ════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the \
sources provided in the CONTEXT block of each message.

Rules:
- Base every statement strictly on the provided context. Do not use outside knowledge, \
and do not guess or infer beyond what the sources state.
- Cite each factual claim with a bracketed source number, e.g. [1] or [2][3], using the \
numbers shown in the context. Place the citation immediately after the claim it supports.
- Only use citation numbers that appear in the context. Never invent a number.
- If the context does not contain enough information to answer, say so explicitly \
(e.g. "The provided sources don't contain an answer to that.") and do not fabricate one.
- If only part of the question is supported, answer that part and clearly state what the \
sources do not cover.
- Format your answer in Markdown: use headings, **bold** for key terms, bullet or numbered \
lists for enumerations, and `code` spans for identifiers, dates, and designations. Keep it \
clean and well-structured, but don't over-format short answers."""


@app.route("/api/chat", methods=["POST"])
def chat():
    user_message = request.json["message"]

    # Retrieve the top-K most relevant chunks, then augment the prompt with a
    # numbered context block so the model can ground its answer and cite sources.
    hits = search(user_message, INDEX, k=10)
    context = "\n\n".join(f"[{i + 1}] {h['text']}" for i, h in enumerate(hits))
    user_content = f"CONTEXT:\n{context}\n\nQUESTION:\n{user_message}"

    # Stream the answer token-by-token over Server-Sent Events so the UI can
    # render it as it arrives. Citations need the *full* answer to extract the
    # [n] markers, so we accumulate the text and emit them in a final event.
    def generate():
        answer_parts: list[str] = []
        with client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        ) as stream:
            for text in stream.text_stream:
                answer_parts.append(text)
                yield f"data: {json.dumps({'type': 'delta', 'text': text})}\n\n"

        answer = "".join(answer_parts)
        citations = _build_citations(answer, hits)
        yield f"data: {json.dumps({'type': 'done', 'citations': citations})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


def _build_citations(answer: str, hits: list[dict]) -> list[dict]:
    """Return one citation entry per unique valid [n] used in the answer.

    Starter implementation: extract the numbers, drop out-of-range, return
    the matching hit's filename + chunk_index. Improve as you like.
    """
    used = [int(n) for n in re.findall(r"\[(\d+)\]", answer)]
    seen: set[int] = set()
    citations: list[dict] = []
    for n in used:
        if n in seen or n < 1 or n > len(hits):
            continue
        seen.add(n)
        h = hits[n - 1]
        citations.append({
            "n": n,
            "source": h["source"],
            "chunk_index": h["chunk_index"],
            "text": h["text"],  # the exact chunk that was cited, for the UI to show
        })
    return citations


if __name__ == "__main__":
    app.run(port=5000, debug=True)
