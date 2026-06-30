import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

// A compact, human-readable summary of a tool call's argument.
function toolArg(input) {
  if (!input) return ''
  if (input.query != null) return `“${input.query}”`
  if (input.section_id != null) return `§ ${input.section_id}`
  return JSON.stringify(input)
}

export default function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')

  async function send(e) {
    e.preventDefault()
    if (!input.trim()) return

    const question = input
    setMessages((m) => [...m, { role: 'user', text: question }])
    setInput('')

    // Add an empty assistant message; we fill it in as tokens stream in.
    setMessages((m) => [
      ...m,
      { role: 'assistant', text: '', citations: [], status: '', tools: [], usage: null, tier: null },
    ])

    const updateLast = (patch) =>
      setMessages((m) => {
        const next = [...m]
        const last = next[next.length - 1]
        next[next.length - 1] = { ...last, ...patch(last) }
        return next
      })

    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: question }),
    })

    // Read the SSE stream: frames are separated by a blank line, each carrying
    // a `data: {json}` line.
    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })

      const frames = buffer.split('\n\n')
      buffer = frames.pop() // keep the trailing partial frame for next read

      for (const frame of frames) {
        const line = frame.split('\n').find((l) => l.startsWith('data:'))
        if (!line) continue
        const payload = JSON.parse(line.slice(5).trim())
        if (payload.type === 'status') {
          // Transient progress line while the agent runs its retrieval tools.
          updateLast(() => ({ status: payload.text }))
        } else if (payload.type === 'tier') {
          // Adaptive-RAG routing decision for this question.
          updateLast(() => ({ tier: { tier: payload.tier, label: payload.label } }))
        } else if (payload.type === 'usage') {
          // Running token totals, refreshed after each model call — including the
          // cache reads/writes that show how much context was served from cache.
          updateLast(() => ({
            usage: {
              input: payload.total_input,
              output: payload.total_output,
              cacheRead: payload.total_cache_read || 0,
              cacheWrite: payload.total_cache_write || 0,
            },
          }))
        } else if (payload.type === 'tool') {
          // Append a persistent record of this tool call (name, args, time, tokens).
          updateLast((last) => ({ tools: [...(last.tools || []), payload], status: '' }))
        } else if (payload.type === 'reset_answer') {
          // The streamed text was the model's preamble before a tool call, not
          // the answer — clear it so only the final answer remains.
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
  }

  return (
    <div className="app">
      <h1>RAG Chat</h1>
      <div className="messages">
        {messages.map((m, i) => (
          <div key={i} className={`msg msg-${m.role}`}>
            <div className="msg-body">
              <b>{m.role}:</b>{' '}
              {m.role === 'assistant'
                ? <div className="markdown"><ReactMarkdown remarkPlugins={[remarkGfm]}>{m.text}</ReactMarkdown></div>
                : m.text}
            </div>
            {m.status && (
              <div className="status working">
                <span className="working-dot" />
                <span>{m.status}</span>
              </div>
            )}
            {(m.usage || m.tier || (m.tools && m.tools.length > 0)) && (
              <div className="telemetry">
                {m.tier && (
                  <div className={`tier tier-${m.tier.tier}`}>
                    <span className="tier-badge">{m.tier.tier}</span>
                    {m.tier.label}
                  </div>
                )}
                {m.usage && (
                  <div className="usage">
                    <span className="usage-item usage-in">
                      ↓ input <b>{m.usage.input.toLocaleString()}</b> tok
                    </span>
                    <span className="usage-item usage-out">
                      ↑ output <b>{m.usage.output.toLocaleString()}</b> tok
                    </span>
                    <span className="usage-item usage-total">
                      Σ <b>{(m.usage.input + m.usage.output).toLocaleString()}</b> tok
                    </span>
                    {m.usage.cacheRead > 0 && (
                      <span className="usage-item usage-cache">
                        ⚡ cached <b>{m.usage.cacheRead.toLocaleString()}</b> tok
                      </span>
                    )}
                  </div>
                )}
                {m.tools && m.tools.length > 0 && (
                  <div className="tools">
                    <div className="tools-label">Tool calls ({m.tools.length})</div>
                    {m.tools.map((t, j) => (
                      <div key={j} className="tool-call">
                        <span className="tool-name">{t.name}</span>
                        <span className="tool-arg">{toolArg(t.input)}</span>
                        <span className="tool-stats">
                          {t.duration_ms.toLocaleString()} ms
                          {' · '}
                          ~{t.result_tokens.toLocaleString()} tok
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
            {m.citations && m.citations.length > 0 && (
              <div className="sources">
                <div className="sources-label">Sources</div>
                {m.citations.map((c) => (
                  <details key={c.n} className="source">
                    <summary>
                      <span className="source-tag">[{c.n}]</span> {c.source}
                      <span className="source-chunk">#{c.chunk_index}</span>
                    </summary>
                    <blockquote className="source-text">{c.text}</blockquote>
                  </details>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
      <form onSubmit={send}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question about the indexed docs..."
          autoFocus
        />
        <button type="submit">Send</button>
      </form>
    </div>
  )
}
