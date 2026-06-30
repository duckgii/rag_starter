"""Context Management RAG starter — indexer.

Walks documents/, chunks each file, embeds chunks, persists the index to disk
so the chat backend can load it without re-indexing.

TODO: implement chunk_text(). The embedding and storage code is provided so
you can focus on the structure.
"""
import pickle
import re
from pathlib import Path

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# Multilingual (50+ languages), 384-dim — same model as the /embedding project.
# Lets the corpus and the queries be in different languages and still match.
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
INDEX_PATH = Path(__file__).parent / "index.pkl"
SECTIONS_PATH = Path(__file__).parent / "sections.pkl"
DOCS_DIR = Path(__file__).parent / "documents"


# ════════════════════════════════════════════════════════════════
# TODO — implement chunk_text
#
# Split `text` into overlapping chunks. A reasonable default:
#   - ~1000 characters per chunk
#   - ~100 characters of overlap
#   - try to break on paragraph boundaries (\n\n) when possible
#
# Return a list of non-empty strings.
# See the lecture slide on chunking for one working implementation.
# ════════════════════════════════════════════════════════════════

def chunk_text(text: str, target_chars: int = 1000, overlap_chars: int = 200) -> list[str]:
    """Split text into overlapping chunks, preferring paragraph boundaries.조

    Greedily packs paragraphs (split on blank lines) into chunks of up to
    ~target_chars. Each chunk carries ~overlap_chars of trailing context from
    the previous chunk so retrieval doesn't lose information across cut points.
    Paragraphs longer than target_chars are split by character window.
    """
    # Split on blank lines, keeping non-empty paragraphs.
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Break up any single paragraph that exceeds the target size.
    pieces: list[str] = []
    for para in paragraphs:
        if len(para) <= target_chars:
            pieces.append(para)
        else:
            for start in range(0, len(para), target_chars):
                pieces.append(para[start:start + target_chars])

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + 2 + len(piece) > target_chars:
            chunks.append(current)
            # Carry the tail of the finished chunk as overlap context.
            tail = current[-overlap_chars:] if overlap_chars > 0 else ""
            current = (tail + "\n\n" + piece) if tail else piece
        else:
            current = (current + "\n\n" + piece) if current else piece

    if current:
        chunks.append(current)

    return [c for c in chunks if c.strip()]


# ════════════════════════════════════════════════════════════════
# Provided: embedding (sentence-transformers, no API key required)
# ════════════════════════════════════════════════════════════════

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Loading embedding model ({MODEL_NAME})... (one-time download ~470MB)")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings. Returns unit-normalized 384-dim vectors."""
    model = get_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


# ════════════════════════════════════════════════════════════════
# Provided: build / save / load / search
# ════════════════════════════════════════════════════════════════

# File types build_index() knows how to read.
SUPPORTED_SUFFIXES = (".md", ".txt", ".pdf")


def read_document(path: Path) -> str:
    """Extract plain text from a supported document.

    `.md` / `.txt` are read directly; `.pdf` is parsed page by page with pypdf
    and the pages are joined with blank lines so chunk_text() can still break on
    paragraph boundaries. Returns "" for a PDF with no extractable text (e.g. a
    scanned/image-only document, which would need OCR).
    """
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)
    return path.read_text()


def build_index() -> list[dict]:
    """Walk DOCS_DIR, chunk each file, embed, return list of records."""
    records: list[dict] = []
    chunk_id = 0
    for path in sorted(DOCS_DIR.glob("*")):
        if path.is_dir() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        text = read_document(path)
        chunks = chunk_text(text)
        if not chunks:
            continue
        vectors = embed(chunks)
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            records.append({
                "chunk_id": chunk_id,
                "source": path.name,
                "chunk_index": i,
                "text": chunk,
                "embedding": vec,
            })
            chunk_id += 1
        print(f"  {path.name}: {len(chunks)} chunks")
    return records


def save_index(records: list[dict]) -> None:
    with INDEX_PATH.open("wb") as f:
        pickle.dump(records, f)


def load_index() -> list[dict]:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"No index found at {INDEX_PATH}. Run `python indexer.py` from the project root first."
        )
    with INDEX_PATH.open("rb") as f:
        return pickle.load(f)


# ════════════════════════════════════════════════════════════════
# Section index — structure-aware retrieval for the agentic retriever
#
# CFR documents have a strong hierarchy: each section ("§ 61.109
# Aeronautical experience.") is a self-contained unit. Indexing whole
# sections (instead of fixed-size chunks) lets the agent read a complete
# regulation and follow the cross-references it cites — which fixed
# chunking can't do, because a section's answer is split across chunks
# and its references point elsewhere in the corpus.
# ════════════════════════════════════════════════════════════════

# A real section header in the body, e.g. "§ 61.109 Aeronautical experience."
# Captures (number, title). Rejects cross-references like "§ 61.107(b)" or
# "§ 61.110 of this part" by requiring a capitalized title word (not a
# connective) right after the number, with the title ending the line.
_SECTION_HEADER = re.compile(
    r"§\s*(\d{1,3}\.\d+[a-z]?)\s+"
    r"(?!of\b|in\b|and\b|or\b|through\b)"
    r"([A-Z][A-Za-z][^§]{1,90}?[.:])"
    r"(?=\s*\n)"
)

# A cross-reference to another section anywhere in a body, e.g. "§ 61.107(b)(1)".
_SECTION_REF = re.compile(r"§+\s*(\d{1,3}\.\d+[a-z]?)")


def split_sections(text: str) -> list[tuple[str, str, str]]:
    """Split a document into (section_id, title, body) tuples at § headers.

    The first occurrence of each section number is treated as its real header
    (later repeats are page running-heads); the body runs to the next header.
    """
    heads: list[tuple[int, str, str]] = []
    seen: set[str] = set()
    for m in _SECTION_HEADER.finditer(text):
        sid = m.group(1)
        if sid in seen:
            continue
        seen.add(sid)
        title = " ".join(m.group(2).rstrip(".:").split())
        heads.append((m.start(), sid, title))

    out: list[tuple[str, str, str]] = []
    for i, (start, sid, title) in enumerate(heads):
        end = heads[i + 1][0] if i + 1 < len(heads) else len(text)
        out.append((sid, title, text[start:end].strip()))
    return out


def find_section_refs(body: str, self_id: str) -> list[str]:
    """Ordered-unique section numbers cited inside `body`, excluding itself."""
    refs: list[str] = []
    for rid in dict.fromkeys(_SECTION_REF.findall(body)):
        if rid != self_id and rid not in refs:
            refs.append(rid)
    return refs


def build_sections() -> list[dict]:
    """Walk DOCS_DIR, split each doc into sections, embed each for navigation."""
    sections: list[dict] = []
    seen_ids: set[str] = set()
    for path in sorted(DOCS_DIR.glob("*")):
        if path.is_dir() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        text = read_document(path)
        secs = [s for s in split_sections(text) if s[0] not in seen_ids]
        if not secs:
            continue
        # Embed "title + opening lines" so navigate_toc ranks on the heading and
        # the section's first lines (which usually state who/what it applies to).
        nav_text = [f"{title}\n{body[:500]}" for _, title, body in secs]
        vectors = embed(nav_text)
        for (sid, title, body), vec in zip(secs, vectors):
            seen_ids.add(sid)
            sections.append({
                "section_id": sid,
                "part": sid.split(".")[0],
                "source": path.name,
                "title": title,
                "text": body,
                "embedding": vec,
            })
        print(f"  {path.name}: {len(secs)} sections")
    return sections


def save_sections(sections: list[dict]) -> None:
    with SECTIONS_PATH.open("wb") as f:
        pickle.dump(sections, f)


def load_sections() -> list[dict]:
    if not SECTIONS_PATH.exists():
        raise FileNotFoundError(
            f"No section index at {SECTIONS_PATH}. Run `python indexer.py` first."
        )
    with SECTIONS_PATH.open("rb") as f:
        return pickle.load(f)


def cosine_distance(a: list[float], b: list[float]) -> float:
    # Both vectors are unit-normalized, so cosine distance == 1 - dot product.
    return 1.0 - sum(x * y for x, y in zip(a, b))


def search(query: str, records: list[dict], k: int = 5) -> list[dict]:
    """Return the k records whose embeddings are closest to the query."""
    [query_vec] = embed([query])
    scored = []
    for r in records:
        distance = cosine_distance(r["embedding"], query_vec)
        scored.append((distance, r))
    scored.sort(key=lambda x: x[0])
    return [r for _, r in scored[:k]]


def main() -> None:
    print(f"Indexing documents from {DOCS_DIR}/")
    records = build_index()
    save_index(records)
    print(f"\n✓ Indexed {len(records)} chunks → {INDEX_PATH.name}")

    print("\nBuilding section index (for the agentic retriever)...")
    sections = build_sections()
    save_sections(sections)
    print(f"✓ Indexed {len(sections)} sections → {SECTIONS_PATH.name}")


if __name__ == "__main__":
    main()
