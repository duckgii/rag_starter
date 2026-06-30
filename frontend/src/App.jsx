import { useState } from 'react'
import ReactMarkdown from 'react-markdown'

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
    setMessages((m) => [...m, { role: 'assistant', text: '', citations: [] }])

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
        if (payload.type === 'delta') {
          updateLast((last) => ({ text: last.text + payload.text }))
        } else if (payload.type === 'done') {
          updateLast(() => ({ citations: payload.citations || [] }))
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
                ? <div className="markdown"><ReactMarkdown>{m.text}</ReactMarkdown></div>
                : m.text}
            </div>
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
