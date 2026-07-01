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
import time

from indexer import cosine_distance, embed, find_section_refs, load_sections

MODEL = "claude-sonnet-4-6"
ROUTER_MODEL = "claude-haiku-4-5"  # cheap model that routes a question by complexity
MAX_ITERS = 6
MAX_SECTION_CHARS = 18000  # cap a single section's text fed back to the model

# Adaptive-RAG: route each question to a retrieval budget matching its complexity,
# instead of always paying for the full agentic loop.
#   A — answer directly, no retrieval         (greetings, out-of-scope questions)
#   B — single-section lookup, no follow_refs  (a specific fact in one section)
#   C — full agentic loop with follow_refs     (multi-section / cross-referential)
TIER_BUDGET = {
    "A": {"max_iters": 0, "follow_refs": False},
    "B": {"max_iters": 2, "follow_refs": False},
    "C": {"max_iters": MAX_ITERS, "follow_refs": True},
}
TIER_LABEL = {
    "A": "💬 Answering directly (no retrieval)",
    "B": "📗 Single-section lookup",
    "C": "📚 Multi-section retrieval",
}

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

def navigate_toc(query: str, k: int = 6) -> list[dict]:
    """Rank sections by similarity of (title + opening) to the query."""
    sections, _ = _load()
    [qv] = embed([query])
    ranked = sorted(sections, key=lambda s: cosine_distance(s["embedding"], qv))[:k]
    return [
        {
            "section_id": s["section_id"],
            "title": s["title"],
            "source": s["source"],
            "preview": " ".join(s["text"][:120].split()),
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
            "Search the regulation table of contents; returns candidate sections "
            "(id, title, short preview) ranked by relevance. Use FIRST. No full "
            "text — call read_section to read one."
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
            "Read one section's FULL text by id (e.g. '61.109'). Returns the text "
            "and the sections it cross-references."
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
            "List the sections a given section cross-references (id, title, "
            "preview). No full text — call read_section on the ones you need."
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


SYSTEM = """You answer U.S. aviation regulation (14 CFR) questions using ONLY \
text you read through the tools. Call tools directly — no preamble or narration.

Steps:
1. navigate_toc(question) → find candidate sections.
2. Pick the RIGHT one from the previews — many share a title (student, private, \
commercial, ATP); match the certificate/rating in the question.
3. read_section to read it in full.
4. If the answer depends on a section it cites, follow_refs to list them, then \
read_section the ones you need.
Stop when you can answer. If the tools don't surface it, say the corpus doesn't \
contain it — don't guess.

Citations: read_section prefixes each section with a number like [3]. Cite every \
factual claim with that bracketed number (e.g. "40 hours [3]"). Never invent a \
number or cite a section you only saw in navigate_toc or follow_refs.

Format in clean Markdown: headings, **bold** terms, and lists for enumerated \
requirements."""


# Routes a question into one Adaptive-RAG tier. Kept tiny so the routing call is
# cheap — it returns a single letter, nothing else.
ROUTER_SYSTEM = """You route questions for a Q&A system over U.S. aviation \
regulations (14 CFR). Classify the user's question into ONE complexity tier and \
reply with only that single letter.

A — No retrieval needed: greetings, small talk, questions about you, or anything \
clearly outside U.S. aviation regulations.
B — Single-section lookup: a specific fact answerable from one regulation section \
(an age, a number, a single requirement) with no need to chase cross-references.
C — Multi-section or cross-referential: the answer likely spans several sections \
or depends on sections that another section points to.

Reply with exactly one character: A, B, or C."""


# Used by tier A, where no regulation lookup is required.
DIRECT_SYSTEM = """You are the assistant for a tool that answers questions about \
U.S. aviation regulations (14 CFR) with citations. The user's message does not \
require looking anything up. Respond briefly and directly: greet them and explain \
what you do if they're saying hello, or politely note that a question is outside \
U.S. aviation regulations if it is."""


def classify(question: str, client) -> tuple[str, object]:
    """Cheaply route a question to a complexity tier. Returns (tier, usage)."""
    resp = client.messages.create(
        model=ROUTER_MODEL,
        max_tokens=5,
        system=ROUTER_SYSTEM,
        messages=[{"role": "user", "content": question}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text").upper()
    tier = next((c for c in text if c in "ABC"), "C")  # default to the safe full path
    return tier, resp.usage


def _cache_last(messages: list[dict]) -> None:
    """Move a single cache breakpoint onto the most recent tool-result message.

    Each ReAct iteration resends the whole history (every section already read),
    so caching the prefix turns that repeated context into cheap cache reads
    instead of re-billed input tokens.
    """
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    block.pop("cache_control", None)
    last = messages[-1].get("content")
    if isinstance(last, list) and last and isinstance(last[-1], dict):
        last[-1]["cache_control"] = {"type": "ephemeral"}


def _status(tool_use) -> str:
    name, inp = tool_use.name, tool_use.input
    if name == "navigate_toc":
        return f"🔎 Searching the table of contents for “{inp.get('query', '')}”…"
    if name == "read_section":
        return f"📖 Reading § {inp.get('section_id', '?')}…"
    if name == "follow_refs":
        return f"🔗 Following cross-references in § {inp.get('section_id', '?')}…"
    return f"… {name}"


def _stream_turn(client, **kwargs):
    """Stream one model turn, yielding text deltas live as they arrive.

    Used as `resp = yield from _stream_turn(...)`: every text delta is forwarded
    to the caller in real time, and the assembled final message (content, usage,
    stop_reason) is returned once the stream completes.
    """
    with client.messages.stream(**kwargs) as stream:
        for event in stream:
            if (event.type == "content_block_delta"
                    and getattr(event.delta, "type", None) == "text_delta"):
                yield {"type": "delta", "text": event.delta.text}
        final = stream.get_final_message()
    return final


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
            # Return previews only (no full text, no citation number) and let the
            # model read_section the ones it actually needs — this keeps the whole
            # referenced corpus out of context unless it's truly required.
            lines = []
            for r in refs:
                rs = get_section(r)
                preview = " ".join(rs["text"][:240].split())
                lines.append(f"§ {rs['section_id']} — {rs['title']}  ·  {preview}")
            return (
                f"§ {s['section_id']} cross-references these sections "
                "(call read_section to read the full text of any you need):\n"
                + "\n".join(lines)
            )
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

    # ── Adaptive-RAG: route the question, then spend only its tier's budget ──
    yield {"type": "status", "text": "🧭 Routing the question…"}
    tier, ru = classify(question, client)
    budget = TIER_BUDGET[tier]

    # Running totals across every model call (router + answer), including the
    # cache reads/writes that prompt caching produces — surfaced so the UI can
    # show how much context was served from cache rather than re-billed.
    total_in = ru.input_tokens
    total_out = ru.output_tokens
    total_cr = getattr(ru, "cache_read_input_tokens", 0) or 0
    total_cw = getattr(ru, "cache_creation_input_tokens", 0) or 0

    def usage_event(step) -> dict:
        return {
            "type": "usage",
            "total_input": total_in,
            "total_output": total_out,
            "total_cache_read": total_cr,
            "total_cache_write": total_cw,
            "step_input": step.input_tokens,
            "step_output": step.output_tokens,
        }

    def done_usage() -> dict:
        return {"total_input": total_in, "total_output": total_out,
                "total_cache_read": total_cr, "total_cache_write": total_cw}

    def add_usage(u) -> None:
        nonlocal total_in, total_out, total_cr, total_cw
        total_in += u.input_tokens
        total_out += u.output_tokens
        total_cr += getattr(u, "cache_read_input_tokens", 0) or 0
        total_cw += getattr(u, "cache_creation_input_tokens", 0) or 0

    yield {"type": "tier", "tier": tier, "label": TIER_LABEL[tier]}
    yield usage_event(ru)

    # Tier A — no retrieval. One direct, tool-free answer, streamed live.
    if budget["max_iters"] == 0:
        yield {"type": "status", "text": "✍️ Writing answer…"}
        resp = yield from _stream_turn(
            client,
            model=MODEL,
            max_tokens=2000,
            system=[{"type": "text", "text": DIRECT_SYSTEM,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": question}],
        )
        add_usage(resp.usage)
        yield usage_event(resp.usage)
        yield {"type": "done", "citations": [], "usage": done_usage()}
        return

    # Tiers B / C — agentic retrieval. Tier B drops follow_refs and caps the loop.
    tools = [t for t in TOOLS if budget["follow_refs"] or t["name"] != "follow_refs"]
    messages: list[dict] = [{"role": "user", "content": question}]
    for step in range(1, budget["max_iters"] + 1):
        _cache_last(messages)  # cache the (growing) section context across iterations
        yield {"type": "status", "text": f"🤔 Thinking… (step {step})"}
        resp = yield from _stream_turn(
            client,
            model=MODEL,
            max_tokens=2000,
            system=[{"type": "text", "text": SYSTEM,
                     "cache_control": {"type": "ephemeral"}}],
            tools=tools,
            messages=messages,
        )
        add_usage(resp.usage)
        yield usage_event(resp.usage)
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason != "tool_use":
            # The final answer already streamed live above; just finalize.
            answer = "".join(b.text for b in resp.content if b.type == "text")
            yield {"type": "done", "citations": citations(answer),
                   "usage": done_usage()}
            return

        # This turn ends in a tool call, so any text streamed above was the
        # model's preamble, not the answer — tell the UI to drop it.
        yield {"type": "reset_answer"}

        results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            yield {"type": "status", "text": _status(block)}
            t0 = time.perf_counter()
            content = run_tool(block.name, block.input)
            duration_ms = round((time.perf_counter() - t0) * 1000)
            # The tool result becomes context on the next call; ~4 chars/token
            # is a good rough estimate of how much it adds.
            result_tokens = round(len(content) / 4)
            # Persistent per-tool record: which tool, its arg, latency, and how
            # much context it produced.
            yield {
                "type": "tool",
                "name": block.name,
                "label": _status(block),
                "input": block.input,
                "duration_ms": duration_ms,
                "result_chars": len(content),
                "result_tokens": result_tokens,
            }
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": content,
            })
        messages.append({"role": "user", "content": results})

    msg = ("I couldn't find enough in the corpus to answer confidently within the "
           "search budget.")
    yield {"type": "delta", "text": msg}
    yield {"type": "done", "citations": citations(msg), "usage": done_usage()}
