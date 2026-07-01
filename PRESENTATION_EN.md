# Agentic RAG over 14 CFR — Presentation Script

> A structure-aware, agentic retrieval system that answers U.S. aviation
> regulation (14 CFR) questions with citations back to the exact source.
> This script follows one flow: **architecture first** (tools → prompt →
> indexing), then **the problems we hit, the options we weighed, and why we
> chose what we chose.**

---

## 0. One-line pitch

> "We built a Q&A assistant over the U.S. federal aviation regulations. Instead
> of the usual 'chop the PDF into chunks and hope the right one ranks first,'
> our agent *navigates the regulation the way a human lawyer would* — searches
> the table of contents, reads whole sections, follows cross-references — and
> cites exactly what it read."

---

## PART 1 — How the system is built

### Slide 1 · The problem & the corpus

- **Domain:** 14 CFR (Title 14, *Code of Federal Regulations* — Aeronautics and Space).
- **Corpus:** 6 official PDFs in `documents/` — Part 61 (pilot certification),
  Part 67 (medical standards), Part 71 (airspace designations), Part 73 (special
  use airspace), Part 91 (operating rules), plus Volume 1.
- **Why it's hard:** the answer to a real question is often (a) split across a
  section, and (b) dependent on *other* sections it cross-references. A naive
  chunk search buries it.

**Talking point:** "Regulations aren't prose — they're a numbered hierarchy.
That structure is the whole game."

---

### Slide 2 · Architecture at a glance

```
 documents/*.pdf
      │
      ▼
 indexer.py ──► sections.pkl   (whole CFR sections + embeddings)
      │        └► index.pkl     (fixed-size chunks — the baseline)
      ▼
 agent.py  ──► ReAct loop: route → navigate_toc → read_section → follow_refs → answer
      │
      ▼
 backend/app.py (Flask)  ──► streams events as Server-Sent Events
      │
      ▼
 frontend (React)  ──► streaming answer, agent trajectory, cost panel, linked citations
```

Four layers: **indexing → agent → backend → frontend.** We'll go through each.

---

### Slide 3 · Indexing — two strategies (`indexer.py`)

We built **two** indexes from the same PDFs:

1. **Chunk index (`index.pkl`) — the baseline.**
   `chunk_text()` splits text into ~1000-char overlapping chunks (200-char
   overlap), preferring paragraph boundaries. This is classic RAG.

2. **Section index (`sections.pkl`) — what the agent uses.**
   `build_sections()` uses a regex to detect real section headers
   (`§ 61.109 Aeronautical experience.`), splits each PDF into **whole
   sections**, and stores `{section_id, part, source, title, text, embedding}`.
   - For ranking, it embeds **`title + first 500 chars`** — the heading plus the
     opening lines that state *who/what the section applies to*.
   - `find_section_refs()` extracts the sections each section cites → enables
     `follow_refs`.

**Embeddings:** `sentence-transformers` multilingual MiniLM, 384-dim,
unit-normalized, **runs locally (no API key)**. Cosine distance = `1 − dot`.

**Talking point:** "Chunking throws away the document's structure. Section
indexing *preserves* it — and structure is exactly what we exploit later."

---

### Slide 4 · The tools (`agent.py`)

The agent has three tools — the verbs a person uses to read a regulation:

| Tool | What it does | Returns |
|---|---|---|
| `navigate_toc(query)` | Semantic search over the section index | Top candidates: id, title, 120-char preview |
| `read_section(id)` | Read one section in full | Full text (capped 18k chars) + its cross-refs |
| `follow_refs(id)` | List the sections a section cites | Previews of referenced sections |

The **ReAct loop**: reason → call a tool → observe the result → repeat, until the
model can answer. It cites every claim with a `[n]` that maps to a section it
actually read.

**Talking point:** "`navigate_toc` finds *where* to look, `read_section` reads,
`follow_refs` walks the citation graph. Three primitives, human workflow."

---

### Slide 5 · The prompt & Adaptive-RAG routing

- **System prompt** tells the model: use ONLY text read via tools; pick the right
  section from previews (many share a title — student/private/commercial/ATP);
  cite with `[n]`; and follow our answer-quality rules (more later).

- **Adaptive-RAG router:** a cheap **Haiku** call classifies each question into a
  complexity tier and returns a single letter:

  | Tier | Meaning | Budget |
  |---|---|---|
  | **A** | No retrieval (greeting / out-of-scope) | 0 iterations, answer directly |
  | **B** | Single-section fact | 2 iterations, no `follow_refs` |
  | **C** | Multi-section / cross-referential | 6 iterations, `follow_refs` on |

**Talking point:** "We don't pay for a 6-step agent to answer 'hello.' The
router spends only the budget the question deserves."

---

### Slide 6 · Cost engineering — prompt caching & streaming

- **Prompt caching:** every ReAct iteration resends the whole growing history
  (every section already read). `_cache_last()` moves a cache breakpoint onto the
  latest message, so that repeated context is billed as **cheap cache reads (0.1×)**
  instead of full input. Write is 1.25×, read is 0.1× — so caching pays off after
  the very first reuse.
- **Streaming:** the backend (Flask) forwards agent events as **Server-Sent
  Events** — `status` (which tool is running), `delta` (answer text), `done`
  (citations + usage). The React UI renders a live trajectory, a token/cost
  panel, and linked sources.

**Talking point:** "The cost panel in the UI is real — we surfaced input / cache
read / cache write per answer so the tradeoffs are visible, not hidden."

---

### Slide 7 · The UI — a glass-box agent (`frontend/src/App.jsx`)

We deliberately designed the interface so the agent's work is **transparent and
verifiable**, not a black box. What we invested in:

- **Live streaming answer.** Tokens render as they arrive (SSE `delta`), so the
  user sees progress instantly instead of waiting for a wall of text.
- **Agent trajectory timeline.** A collapsible step list shows the whole run: the
  routing decision (→ tier), every tool call with its argument, its latency, and
  how many tokens the result added — then the final answer. You can see *how* the
  agent reached its conclusion.
- **Token / cost panel.** Per answer: input vs output tokens, a stacked bar of
  billed input / cache read / cache write, the dollar cost, cache-savings %, and
  total context size. The economics are on screen, not hidden.
- **Verifiable citations.** Inline `[n]` markers are clickable — they scroll to
  the matching source, expand it, and briefly **flash-highlight** it. Each source
  shows a readable document title and links out to eCFR + the govinfo source PDF.
- **Live status banner.** An animated pulse shows the current step ("Reading
  § 61.109…", "Following cross-references…").
- **Preamble suppression.** If a streamed turn ends in a tool call, that text was
  just the model thinking aloud — we drop it (`reset_answer`) so only the real
  answer remains.
- **Always-available examples.** A sticky right sidebar (numbered 1–5, disabled
  while busy, responsive to a stacked layout on narrow screens) lets you fire a
  sample question at any time.
- **Stop button** aborts a run mid-stream; **Copy answer** exports the full
  Markdown with sources.

**Talking point:** "For an agent, trust comes from transparency. Every answer
shows *what it did, what it cost, and where it came from* — the trajectory, the
cost panel, and clickable citations are all one idea: no black box."

---

## PART 2 — Problems, options, and what we chose

> For each problem: what broke, the options we compared, their pros/cons, and our
> decision — grounded in live measurements.

### Slide 8 · Problem 1 — Chunk retrieval buries the answer

- **Symptom:** with fixed-size chunks, the correct section ranks ~17th among
  near-identical chunks, and its content is split across several of them.
- **Options:**
  - *(a)* Bigger chunks / more k → still no structure, more noise.
  - *(b)* **Section-level index + agentic navigation.** ✅
- **Tradeoff:** the agent costs more per query than one-shot retrieval, but it can
  read a *complete* rule and follow references — which chunking fundamentally
  can't.
- **Chosen:** *(b)* — this is the core architecture (commit `0020ae7`).

---

### Slide 9 · Problem 2 — One-size budget is wasteful

- **Symptom:** running the full agentic loop for every message (even a greeting)
  wastes tokens and latency.
- **Options:**
  - *(a)* Always run the full loop → simple, expensive.
  - *(b)* **Adaptive-RAG tiered routing (A/B/C).** ✅
- **Tradeoff:** routing adds one cheap Haiku call and a mis-routing risk — we
  found Tier B's 2-iteration budget is tight (navigate → read → answer needs 3
  turns). Mitigated by the fallback in Problem 3.
- **Chosen:** *(b)* — complexity-matched budgets.

---

### Slide 10 · Problem 3 — Aggregation questions fail  ⭐ the main story

**Case study — Q2:** *"Which medical conditions disqualify a first-class airman
medical certificate?"*

- The answer spans **6 sibling sections** (§67.103 Eye, .105 Ear, .107 Mental,
  .109 Neurologic, .111 Cardiovascular, .113 General).
- `navigate_toc`'s top-6 surfaced only **2–3** of them, buried behind
  near-identical decoys from the **second- and third-class** blocks
  (§67.2xx / §67.3xx — same titles!).
- The agent re-queried three times, exhausted its 6-iteration budget, and **gave
  up with no answer** — ~$0.036 spent for nothing.

**Three options we compared:**

| Option | Idea | Pro | Con |
|---|---|---|---|
| **1. Forced synthesis** | On budget-exhaustion, answer with whatever was read instead of giving up | Cheap; kills the "no answer" case | Doesn't improve coverage — §67.113 still missing |
| **2. Recall expansion (chosen core)** | `navigate_toc` pulls the *whole dominant numbering block* (§67.1xx) and drops off-block decoys | Fixes coverage **and** precision; linear cost | Needs numbering-structure awareness + a size cap |
| **3. Iterative deepening** | On failure, double the budget 2→4→8→16 and retry | Simple to implement | Exponential cost; doesn't fix the root cause (low recall) — just lets it thrash longer |

**Key insight for Option 2:** CFR numbering *encodes* structure — §67.**1**xx =
first-class, §67.**2**xx = second, §67.**3**xx = third. So when hits cluster in
one small block, pull the *entire block* and exclude the other classes. Gated to
tight blocks (`≤ 14` sections) so big subparts (Part 91's ~48 §91.1xx sections)
are never swept in — verified **no regression** on the other four questions.

**What we chose:** **Option 2 as the fix + Option 1 as a safety net.** We
rejected iterative deepening — it grows cost without addressing *why* retrieval
missed.

**Measured result (Q2, live):**

| | Before | Forced synthesis only | **Approach 2 (final)** |
|---|---|---|---|
| Outcome | Gave up, no answer | Incomplete (5/6, flags gap) | **Complete (6/6)** |
| Sections cited | 0 | 5 | **6** |
| Cost | ~$0.036 | ~$0.080 | **~$0.057** |

→ A **complete** answer for **less** than the incomplete one — because expansion
stops the agent from thrashing.

---

### Slide 11 · Problem 4 — Citations were unreadable & unverifiable

- **Symptom:** PDF text has hard line-wraps, words hyphenated across breaks
  (`un-\nless`), and page-footer noise; the UI showed a raw filename with no way
  to check the source.
- **Solution (chosen):**
  - Clean the text (stitch hyphenation, strip footers, collapse wraps) and split
    it into paragraphs at subsection markers `(a)/(b)/(1)`.
  - Show a **human-readable document title** ("14 CFR Part 67 — Medical Standards
    and Certification").
  - Add two links: **eCFR** (jumps to the exact section) and **the govinfo source
    PDF** (the exact document we indexed — the filename maps 1:1 to the govinfo
    granule; both verified to resolve).
  - "Copy answer" exports the whole thing — answer + sourced blockquotes — as
    Markdown.

---

### Slide 12 · Problem 5 — Answer quality (jargon, structure, verbosity)

- **Symptom:** unexplained abbreviations, headings forced onto one-line answers,
  restating the question.
- **Solution (chosen):** explicit prompt rules —
  - **Define each abbreviation once, abbreviation-first:** `14 CFR (Code of
    Federal Regulations)`, `FFS (full flight simulator)` — *never* reversed; then
    use the short form.
  - **Match structure to the question:** a sentence for a single fact, a list for
    enumerated requirements, a **table only for comparisons.**
  - **No filler:** lead with the answer, drop irrelevant parts.
- **Interesting finding:** the phrase *"including citation ones like CFR"* is
  load-bearing — without it the model treats "14 CFR" as a bare citation and
  skips the expansion. We verified this with live runs, then trimmed the prompt
  while keeping that one phrase.

**Talking point:** "Prompt engineering here was empirical — we changed one line,
ran the question live, and watched what actually happened."

---

### Slide 13 · Results & takeaways

- **Structure beats chunking** for hierarchical documents — index and navigate by
  the document's own units.
- **Match the effort to the question** (Adaptive-RAG) to control cost.
- **Fix the root cause, not the symptom:** for the aggregation failure, recall
  expansion (address *why* retrieval missed) beat iterative deepening (just spend
  more). We layered a cheap safety net on top.
- **Make provenance first-class:** clean, paragraph-split, *linked* citations turn
  a demo into something you can trust.
- **Transparency is a feature:** the trajectory, live cost panel, and clickable
  citations make the agent a glass box — you can see and verify every answer.
- **Prompt rules, verified live**, not guessed.

**Closing line:** "The theme throughout: *use the structure of the data.* It
drove the indexing, the tools, the retrieval fix, and even the citation links."

---

### Appendix · Key files & knobs

- `indexer.py` — `chunk_text`, `build_sections`, `split_sections`,
  `find_section_refs`, `embed` (local MiniLM, 384-dim).
- `agent.py` — tools, `TIER_BUDGET` (A/B/C), `navigate_toc` + block expansion
  (`TOC_EXPAND_GROUP_MAX = 14`), forced-synthesis fallback, `_cache_last`.
- `backend/app.py` — Flask SSE endpoint `/api/chat`.
- `frontend/src/App.jsx` — streaming UI, trajectory, cost panel, linked sources.
- Models: answer = `claude-sonnet-4-6`, router = `claude-haiku-4-5`.
- Constants: `MAX_ITERS = 6`, `MAX_SECTION_CHARS = 18000`, `navigate_toc k = 6`.
