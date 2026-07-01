import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

// Sonnet 4.6 pricing ($/1M tokens). Cache read = 0.1×, cache write = 1.25×.
// (Router runs on cheaper Haiku, but its tokens are tiny — we approximate.)
const PRICE = { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 }

const TIER_DESC = { A: '직답 (검색 없음)', B: '단일 섹션 조회', C: '다중 섹션 검색' }

const EXAMPLES = [
  { q: '이 봇은 무슨 일을 하나요?', tier: 'A' },
  { q: '자가용 조종사 자격의 최소 나이는?', tier: 'B' },
  { q: 'VFR 연료 예비 요구량은? 비행기와 회전익기를 비교해줘', tier: 'C' },
]

// A compact, human-readable summary of a tool call's argument.
function toolArg(input) {
  if (!input) return ''
  if (input.query != null) return `“${input.query}”`
  if (input.section_id != null) return `§ ${input.section_id}`
  return JSON.stringify(input)
}

function stepIcon(name) {
  return { navigate_toc: '🔎', read_section: '📖', follow_refs: '🔗' }[name] || '🔧'
}

// Turn inline citation markers like [1] into links that jump to the matching
// Sources entry below. Scoped by message index so numbers don't collide.
function linkifyCitations(text, msgIndex) {
  return text.replace(/\[(\d+)\](?!\()/g, (_, n) => `[[${n}]](#cite-${msgIndex}-${n})`)
}

// Custom markdown link renderer: citation links (#cite-…) scroll to the source,
// open it if collapsed, and briefly highlight it. Other links render normally.
const mdComponents = {
  a({ href, children, node, ...props }) {
    if (href && href.startsWith('#cite-')) {
      const onClick = (e) => {
        e.preventDefault()
        const el = document.getElementById(href.slice(1))
        if (!el) return
        el.open = true
        el.scrollIntoView({ behavior: 'smooth', block: 'center' })
        el.classList.add('cite-flash')
        setTimeout(() => el.classList.remove('cite-flash'), 1200)
      }
      return (
        <a href={href} className="cite-link" onClick={onClick} {...props}>
          {children}
        </a>
      )
    }
    return <a href={href} {...props}>{children}</a>
  },
}

// Derived token/cost figures for one answer.
function costOf(u) {
  const dollars =
    (u.input * PRICE.input + u.output * PRICE.output +
     u.cacheRead * PRICE.cacheRead + u.cacheWrite * PRICE.cacheWrite) / 1e6
  const actualInput = u.input * PRICE.input + u.cacheRead * PRICE.cacheRead + u.cacheWrite * PRICE.cacheWrite
  const noCacheInput = (u.input + u.cacheRead + u.cacheWrite) * PRICE.input
  const savingsPct = noCacheInput > 0 ? Math.round((1 - actualInput / noCacheInput) * 100) : 0
  const totalCtx = u.input + u.cacheRead + u.cacheWrite
  return { dollars, savingsPct, totalCtx }
}

// ── Sub-components ───────────────────────────────────────────────

function WorkingBanner({ text }) {
  return (
    <div className="status working">
      <span className="working-dot" />
      <span>{text}</span>
    </div>
  )
}

function Timeline({ m }) {
  const steps = []
  if (m.tier) steps.push({ icon: '🧭', title: `Route → ${m.tier.tier} tier`, meta: TIER_DESC[m.tier.tier] })
  for (const t of m.tools || []) {
    steps.push({
      icon: stepIcon(t.name),
      title: `${t.name} ${toolArg(t.input)}`,
      meta: `${t.duration_ms.toLocaleString()}ms · ~${t.result_tokens.toLocaleString()} tok`,
    })
  }
  if (m.text && !m.status) {
    steps.push({ icon: '✍️', title: 'Answer', meta: m.usage ? `${m.usage.output.toLocaleString()} tok` : '' })
  }
  if (steps.length === 0) return null
  return (
    <details className="timeline-wrap" open>
      <summary className="timeline-title">Agent trajectory · {steps.length} steps</summary>
      <ol className="timeline">
        {steps.map((s, k) => (
          <li key={k} className="tl-step">
            <span className="tl-icon">{s.icon}</span>
            <span className="tl-body">
              <span className="tl-title">{s.title}</span>
              {s.meta && <span className="tl-meta">{s.meta}</span>}
            </span>
          </li>
        ))}
      </ol>
    </details>
  )
}

function CostPanel({ usage }) {
  const c = costOf(usage)
  const totalInput = c.totalCtx // uncached input + cache read + cache write
  const ctx = totalInput || 1
  const pct = (n) => `${(n / ctx) * 100}%`
  return (
    <div className="cost">
      <div className="cost-io">
        <span className="io io-in">↓ input <b>{usage.input.toLocaleString()}</b> tok</span>
        <span className="io io-out">↑ output <b>{usage.output.toLocaleString()}</b> tok</span>
      </div>
      <div className="cost-bar" title="전체 컨텍스트 구성 (billed input · cache read · cache write)">
        <span className="seg seg-in" style={{ width: pct(usage.input) }} />
        <span className="seg seg-read" style={{ width: pct(usage.cacheRead) }} />
        <span className="seg seg-write" style={{ width: pct(usage.cacheWrite) }} />
      </div>
      <div className="cost-legend">
        <span><i className="dot dot-in" />billed input {usage.input.toLocaleString()}</span>
        <span><i className="dot dot-read" />⚡cache {usage.cacheRead.toLocaleString()}</span>
        <span><i className="dot dot-write" />write {usage.cacheWrite.toLocaleString()}</span>
      </div>
      <div className="cost-summary">
        <span className="cost-dollar">≈ ${c.dollars.toFixed(5)}</span>
        {c.savingsPct > 0 && <span className="cost-save">캐시로 {c.savingsPct}% 절감</span>}
        <span className="cost-ctx">📦 context {totalInput.toLocaleString()} tok</span>
      </div>
    </div>
  )
}

function Sources({ citations, idx }) {
  return (
    <div className="sources">
      <div className="sources-label">Sources</div>
      {citations.map((c) => (
        <details key={c.n} id={`cite-${idx}-${c.n}`} className="source">
          <summary>
            <span className="source-tag">[{c.n}]</span> {c.source}
            <span className="source-chunk">#{c.chunk_index}</span>
          </summary>
          <blockquote className="source-text">{c.text}</blockquote>
        </details>
      ))}
    </div>
  )
}

// ── App ──────────────────────────────────────────────────────────

export default function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const bottomRef = useRef(null)
  const abortRef = useRef(null)

  // Keep the view pinned to the latest content while streaming.
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ block: 'end' })
  }, [messages])

  async function send(question) {
    if (!question.trim() || busy) return
    setError('')
    setMessages((m) => [...m, { role: 'user', text: question }])
    setInput('')
    setMessages((m) => [
      ...m,
      { role: 'assistant', text: '', citations: [], status: '', tools: [], usage: null, tier: null, error: '' },
    ])

    const updateLast = (patch) =>
      setMessages((m) => {
        const next = [...m]
        const last = next[next.length - 1]
        next[next.length - 1] = { ...last, ...patch(last) }
        return next
      })

    setBusy(true)
    const controller = new AbortController()
    abortRef.current = controller

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: question }),
        signal: controller.signal,
      })
      if (!res.ok) throw new Error(`서버 오류 ${res.status}`)

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })

        const frames = buffer.split('\n\n')
        buffer = frames.pop()

        for (const frame of frames) {
          const line = frame.split('\n').find((l) => l.startsWith('data:'))
          if (!line) continue
          const payload = JSON.parse(line.slice(5).trim())
          if (payload.type === 'status') {
            updateLast(() => ({ status: payload.text }))
          } else if (payload.type === 'tier') {
            updateLast(() => ({ tier: { tier: payload.tier, label: payload.label } }))
          } else if (payload.type === 'usage') {
            updateLast(() => ({
              usage: {
                input: payload.total_input,
                output: payload.total_output,
                cacheRead: payload.total_cache_read || 0,
                cacheWrite: payload.total_cache_write || 0,
              },
            }))
          } else if (payload.type === 'tool') {
            updateLast((last) => ({ tools: [...(last.tools || []), payload], status: '' }))
          } else if (payload.type === 'reset_answer') {
            updateLast(() => ({ text: '' }))
          } else if (payload.type === 'delta') {
            updateLast((last) => ({ text: last.text + payload.text, status: '' }))
          } else if (payload.type === 'done') {
            updateLast((last) => ({
              citations: payload.citations || [],
              status: '',
              usage: payload.usage
                ? {
                    input: payload.usage.total_input,
                    output: payload.usage.total_output,
                    cacheRead: payload.usage.total_cache_read || 0,
                    cacheWrite: payload.usage.total_cache_write || 0,
                  }
                : last.usage,
            }))
          }
        }
      }
    } catch (err) {
      if (err.name === 'AbortError') {
        updateLast((last) => ({ status: '', text: last.text || '_(중단됨)_' }))
      } else {
        setError(err.message || '요청 실패')
        updateLast(() => ({ status: '', error: err.message || '요청 실패' }))
      }
    } finally {
      setBusy(false)
      abortRef.current = null
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>RAG Chat</h1>
        <span className="app-sub">14 CFR · agentic retrieval</span>
      </header>

      <div className="messages">
        {messages.length === 0 && (
          <div className="empty">
            <p className="empty-lead">항공 규정(14 CFR)에 대해 물어보세요. 아래 예시를 눌러 시작해도 됩니다:</p>
            <div className="examples">
              {EXAMPLES.map((ex, k) => (
                <button key={k} className="example" onClick={() => send(ex.q)}>
                  <span className={`tier-badge tier-badge-${ex.tier}`} title={TIER_DESC[ex.tier]}>{ex.tier}</span>
                  <span>{ex.q}</span>
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i} className={`row row-${m.role}`}>
            <div className="avatar">{m.role === 'user' ? '🧑' : '🤖'}</div>
            <div className="bubble">
              {m.role === 'assistant' ? (
                <div className="markdown">
                  <ReactMarkdown remarkPlugins={[remarkGfm]} components={mdComponents}>
                    {linkifyCitations(m.text, i)}
                  </ReactMarkdown>
                </div>
              ) : (
                <div className="user-text">{m.text}</div>
              )}

              {m.status && <WorkingBanner text={m.status} />}
              {m.error && <div className="error-inline">⚠️ {m.error}</div>}

              {m.role === 'assistant' && (m.tier || (m.tools && m.tools.length > 0)) && <Timeline m={m} />}
              {m.usage && <CostPanel usage={m.usage} />}
              {m.citations && m.citations.length > 0 && <Sources citations={m.citations} idx={i} />}

              {m.role === 'assistant' && m.text && !m.status && (
                <button className="copy-btn" onClick={() => navigator.clipboard?.writeText(m.text)}>
                  답변 복사
                </button>
              )}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {error && <div className="error-banner">⚠️ {error}</div>}

      <form onSubmit={(e) => { e.preventDefault(); send(input) }}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="항공 규정에 대해 질문하세요..."
          autoFocus
          disabled={busy}
        />
        {busy ? (
          <button type="button" className="stop-btn" onClick={() => abortRef.current?.abort()}>
            ■ Stop
          </button>
        ) : (
          <button type="submit">Send</button>
        )}
      </form>
    </div>
  )
}
