"""Agentic (ReAct) retriever over the CFR section index.

Three tools let the model navigate regulations the way a person would, instead
of relying on fixed-size chunk similarity (which buries the answer — the right
section ranks ~17th among near-identical chunks and is split across several):

  1. navigate_toc(query)   — find candidate sections from the table of contents
  2. read_section(id)      — read one section in full
  3. follow_refs(id)       — read every section that section cross-references

The model loops (reason → act → observe) until it has enough to answer, then
cites the sections it actually used. A hard iteration cap stops it from looping
when the corpus simply doesn't contain the answer.
"""
import re

from indexer import cosine_distance, embed, find_section_refs, load_sections

MODEL = "claude-sonnet-4-6"
MAX_ITERS = 6
MAX_SECTION_CHARS = 18000  # cap a single section's text fed back to the model

# Lazily loaded so importing this module (and starting the server) stays fast.
_SECTIONS: list[dict] | None = None
_BY_ID: dict[str, dict] | None = None


def _load():
    global _SECTIONS, _BY_ID
    if _SECTIONS is None:
        _SECTIONS = load_sections()
        _BY_ID = {s["section_id"]: s for s in _SECTIONS}
    return _SECTIONS, _BY_ID


# ── Tool implementations ────────────────────────────────────────────

def navigate_toc(query: str, k: int = 12) -> list[dict]:
    """Rank sections by similarity of (title + opening) to the query."""
    sections, _ = _load()
    [qv] = embed([query])
    ranked = sorted(sections, key=lambda s: cosine_distance(s["embedding"], qv))[:k]
    return [
        {
            "section_id": s["section_id"],
            "title": s["title"],
            "source": s["source"],
            "preview": " ".join(s["text"][:240].split()),
        }
        for s in ranked
    ]


def get_section(section_id: str) -> dict | None:
    _, by_id = _load()
    return by_id.get(section_id)


# ── Anthropic tool schemas ──────────────────────────────────────────

TOOLS = [
    {
        "name": "navigate_toc",
        "description": (
            "Search the regulation table of contents and return candidate "
            "sections (id, title, and a short preview) ranked by relevance. "
            "Use this FIRST to locate which sections matter. It returns NO full "
            "text — call read_section to actually read one."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What you're looking for, in natural language.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_section",
        "description": (
            "Read the FULL text of one section by its id (e.g. '61.109'). "
            "Returns the section text and lists the other sections it "
            "cross-references."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "section_id": {
                    "type": "string",
                    "description": "Section number without the § sign, e.g. '61.109'.",
                }
            },
            "required": ["section_id"],
        },
    },
    {
        "name": "follow_refs",
        "description": (
            "Read the full text of every section cross-referenced by the given "
            "section. Use after read_section when the answer depends on a "
            "section the text points to (e.g. 'as listed in § 61.107(b)(1)')."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "section_id": {
                    "type": "string",
                    "description": "The section whose cross-references to read.",
                }
            },
            "required": ["section_id"],
        },
    },
]


SYSTEM = """You answer questions about U.S. aviation regulations (14 CFR) using \
ONLY what you read through the tools. You cannot see any regulation text until \
you read it.

Strategy:
1. Call navigate_toc with the user's question to find candidate sections.
2. Use the previews to pick the RIGHT one — many sections share a title (e.g. \
several "Aeronautical experience" sections exist for student, private, \
commercial, and ATP certificates). Match the certificate/rating in the question.
3. Call read_section on it to read the full text.
4. If the section's answer depends on another section it cites (e.g. "areas of \
operation listed in § 61.107(b)(1)"), call follow_refs to read those too.
5. Repeat as needed. Stop once you can answer — or, if the tools don't surface \
the information, say the corpus doesn't contain it rather than guessing.

Citation rules:
- Every section returned by read_section / follow_refs is prefixed with a \
number like [3]. Cite each factual claim with that bracketed number, placed \
right after the claim, e.g. "at least 40 hours of flight time [3]".
- Only use numbers that were shown to you. Never invent one or cite a section \
you only saw in navigate_toc (you didn't read it).
- If the sources don't answer the question, say so explicitly and don't fabricate.

Format the answer in clean Markdown: headings, **bold** key terms, and bullet \
or numbered lists for enumerated requirements."""


def _status(tool_use) -> str:
    name, inp = tool_use.name, tool_use.input
    if name == "navigate_toc":
        return f"🔎 Searching the table of contents for “{inp.get('query', '')}”…"
    if name == "read_section":
        return f"📖 Reading § {inp.get('section_id', '?')}…"
    if name == "follow_refs":
        return f"🔗 Following cross-references in § {inp.get('section_id', '?')}…"
    return f"… {name}"


def _chunked(text: str, size: int = 60):
    for i in range(0, len(text), size):
        yield text[i:i + size]


def run(question: str, client):
    """Drive the ReAct loop, yielding SSE-ready event dicts.

    Yields {"type": "status"|"delta"|"done", ...}. Citation numbers are assigned
    to sections the moment a tool returns their text, so the numbers the model
    cites map back to real sections.
    """
    cite_list: list[dict] = []          # index i -> section shown as [i+1]
    cite_n: dict[str, int] = {}         # section_id -> citation number

    def cite(section: dict) -> int:
        sid = section["section_id"]
        if sid not in cite_n:
            cite_list.append(section)
            cite_n[sid] = len(cite_list)
        return cite_n[sid]

    def render(section: dict) -> str:
        n = cite(section)
        body = section["text"]
        trunc = "" if len(body) <= MAX_SECTION_CHARS else "\n…[section truncated]"
        refs = find_section_refs(section["text"], section["section_id"])
        refs = [r for r in refs if get_section(r)]
        refline = f"\nCross-references: {', '.join('§ ' + r for r in refs)}" if refs else ""
        header = f"[{n}] § {section['section_id']} — {section['title']} ({section['source']})"
        return f"{header}{refline}\n\n{body[:MAX_SECTION_CHARS]}{trunc}"

    def run_tool(name: str, inp: dict) -> str:
        if name == "navigate_toc":
            cands = navigate_toc(inp.get("query", ""))
            if not cands:
                return "No candidate sections found."
            lines = [
                f"§ {c['section_id']} — {c['title']}  ·  {c['preview']}" for c in cands
            ]
            return "Candidate sections (call read_section to read one):\n" + "\n".join(lines)
        if name == "read_section":
            s = get_section(inp.get("section_id", ""))
            return render(s) if s else f"No section § {inp.get('section_id')} in the corpus."
        if name == "follow_refs":
            s = get_section(inp.get("section_id", ""))
            if not s:
                return f"No section § {inp.get('section_id')} in the corpus."
            refs = [r for r in find_section_refs(s["text"], s["section_id"]) if get_section(r)]
            if not refs:
                return f"§ {s['section_id']} cites no other sections in this corpus."
            return "\n\n———\n\n".join(render(get_section(r)) for r in refs)
        return f"Unknown tool: {name}"

    def citations(answer: str) -> list[dict]:
        used = [int(n) for n in re.findall(r"\[(\d+)\]", answer)]
        seen: set[int] = set()
        out: list[dict] = []
        for n in used:
            if n in seen or n < 1 or n > len(cite_list):
                continue
            seen.add(n)
            s = cite_list[n - 1]
            out.append({
                "n": n,
                "source": s["source"],
                "chunk_index": f"§ {s['section_id']}",
                "text": s["text"][:900],
            })
        return out

    messages: list[dict] = [{"role": "user", "content": question}]
    for _ in range(MAX_ITERS):
        resp = client.messages.create(
            model=MODEL,
            max_tokens=2000,
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason != "tool_use":
            answer = "".join(b.text for b in resp.content if b.type == "text")
            for piece in _chunked(answer):
                yield {"type": "delta", "text": piece}
            yield {"type": "done", "citations": citations(answer)}
            return

        results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            yield {"type": "status", "text": _status(block)}
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": run_tool(block.name, block.input),
            })
        messages.append({"role": "user", "content": results})

    msg = ("I couldn't find enough in the corpus to answer confidently within the "
           "search budget.")
    yield {"type": "delta", "text": msg}
    yield {"type": "done", "citations": citations(msg)}
