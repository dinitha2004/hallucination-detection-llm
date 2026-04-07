/**
 * QueryInput.jsx — User Prompt Input Component
 * Left panel: textarea + submit button + max tokens slider
 *
 * Author: Chalani Dinitha (20211032)
 */

import { useState } from 'react'
import { Send, Loader2 } from 'lucide-react'

const EXAMPLE_PROMPTS = [
  'The capital of France is',
  'Einstein was born in 1879 in Germany',
  'Shakespeare wrote Hamlet in approximately',
  'NASA was founded on October 1,',
  'The speed of light is approximately',
]

export default function QueryInput({ onSubmit, loading, disabled }) {
  const [prompt,    setPrompt]    = useState('')
  const [maxTokens, setMaxTokens] = useState(30)

  const handleSubmit = () => {
    if (!prompt.trim() || loading || disabled) return
    onSubmit(prompt.trim(), maxTokens)
  }

  const handleKey = (e) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) handleSubmit()
  }

  return (
    <div className="space-y-4">
      <div>
        <label className="block text-xs font-semibold text-slate-400
                           uppercase tracking-wider mb-2">
          Prompt
        </label>
        <textarea
          value={prompt}
          onChange={e => setPrompt(e.target.value)}
          onKeyDown={handleKey}
          placeholder="Enter a factual question or statement…"
          rows={5}
          disabled={disabled || loading}
          className="w-full rounded-xl border border-slate-700 bg-slate-900
                     text-slate-100 placeholder-slate-600 font-mono text-sm
                     p-3 resize-none focus:outline-none focus:ring-2
                     focus:ring-brand-500/50 focus:border-brand-500/50
                     transition disabled:opacity-40"
        />
        <p className="text-xs text-slate-600 font-mono mt-1">
          ⌘+Enter to submit
        </p>
      </div>

      {/* Max tokens slider */}
      <div>
        <label className="block text-xs font-semibold text-slate-400
                           uppercase tracking-wider mb-2">
          Max new tokens: <span className="text-brand-500">{maxTokens}</span>
        </label>
        <input
          type="range"
          min={10} max={100} step={5}
          value={maxTokens}
          onChange={e => setMaxTokens(Number(e.target.value))}
          className="w-full accent-brand-500"
          disabled={disabled || loading}
        />
        <div className="flex justify-between text-xs text-slate-600 font-mono mt-1">
          <span>10</span><span>100</span>
        </div>
      </div>

      {/* Submit button */}
      <button
        onClick={handleSubmit}
        disabled={!prompt.trim() || loading || disabled}
        className="w-full flex items-center justify-center gap-2 rounded-xl
                   bg-brand-500 hover:bg-brand-600 active:bg-brand-700
                   text-white font-semibold text-sm py-3 transition
                   disabled:opacity-40 disabled:cursor-not-allowed"
      >
        {loading ? (
          <><Loader2 size={16} className="animate-spin" /> Analysing…</>
        ) : (
          <><Send size={16} /> Detect Hallucination</>
        )}
      </button>

      {/* Example prompts */}
      <div>
        <p className="text-xs font-semibold text-slate-500 uppercase
                      tracking-wider mb-2">
          Examples
        </p>
        <div className="space-y-1">
          {EXAMPLE_PROMPTS.map(p => (
            <button
              key={p}
              onClick={() => setPrompt(p)}
              disabled={loading}
              className="w-full text-left text-xs font-mono text-slate-500
                         hover:text-brand-400 hover:bg-slate-800/50 rounded-lg
                         px-2 py-1.5 transition truncate disabled:opacity-40"
            >
              {p}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
