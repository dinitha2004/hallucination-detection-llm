/**
 * TokenDisplay.jsx — Token-Level Highlighting Component
 * This is the CORE visual output of your research (Gap 2).
 *
 * Renders each token with colour coding:
 *   RED    = hallucinated EAT token (score ≥ 0.65)
 *   YELLOW = suspicious EAT token   (score 0.45–0.65)
 *   NORMAL = safe token
 *
 * Hover over any token to see its score as a tooltip.
 *
 * Author: Chalani Dinitha (20211032)
 */

import { useState } from 'react'

function getRiskClass(token) {
  if (token.is_flagged)                              return 'token-hallucinated cursor-help'
  if (token.risk_level === 'suspicious' && token.is_eat) return 'token-suspicious cursor-help'
  return 'token-safe text-slate-200'
}

function TokenChip({ token }) {
  const [show, setShow] = useState(false)
  const cls = getRiskClass(token)
  const hasInfo = token.is_eat || token.hallucination_score > 0.1

  return (
    <span className="relative inline-block">
      <span
        className={`font-mono text-sm ${cls} whitespace-pre-wrap`}
        onMouseEnter={() => hasInfo && setShow(true)}
        onMouseLeave={() => setShow(false)}
      >
        {token.token}
      </span>

      {/* Tooltip */}
      {show && (
        <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1.5 z-50
                          whitespace-nowrap rounded-lg border border-slate-700
                          bg-slate-900 px-2.5 py-1.5 text-xs font-mono
                          shadow-xl pointer-events-none">
          <span className="block text-slate-300">
            score: <span className={
              token.hallucination_score >= 0.65 ? 'text-red-400 font-semibold' :
              token.hallucination_score >= 0.45 ? 'text-yellow-400' :
              'text-green-400'
            }>
              {token.hallucination_score.toFixed(3)}
            </span>
          </span>
          {token.is_eat && (
            <span className="block text-brand-400">
              EAT · {token.entity_type || 'entity'}
            </span>
          )}
          <span className="block text-slate-500">{token.risk_level}</span>
        </span>
      )}
    </span>
  )
}

export default function TokenDisplay({ result }) {
  if (!result) return null

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">
          Generated Response
        </h2>
        <div className="flex items-center gap-3 text-xs font-mono text-slate-500">
          <span>{result.annotated_tokens.length} tokens</span>
          <span>·</span>
          <span>{result.num_eat_tokens} EATs</span>
          <span>·</span>
          <span>{result.processing_time_ms.toFixed(0)}ms</span>
        </div>
      </div>

      {/* Token rendering — Gap 2 visualisation */}
      <div className="leading-8 tracking-wide">
        {result.annotated_tokens.map((token, i) => (
          <TokenChip key={i} token={token} />
        ))}
      </div>

      {/* EAT summary */}
      {result.num_eat_tokens > 0 && (
        <div className="pt-2 border-t border-slate-800">
          <p className="text-xs font-mono text-slate-500 mb-1.5">
            Exact Answer Tokens detected:
          </p>
          <div className="flex flex-wrap gap-1.5">
            {result.annotated_tokens
              .filter(t => t.is_eat)
              .map((t, i) => (
                <span
                  key={i}
                  className={`text-xs font-mono px-2 py-0.5 rounded-md border ${
                    t.is_flagged
                      ? 'border-red-500/40 bg-red-500/10 text-red-400'
                      : t.risk_level === 'suspicious'
                      ? 'border-yellow-500/40 bg-yellow-500/10 text-yellow-400'
                      : 'border-slate-700 bg-slate-800 text-slate-400'
                  }`}
                >
                  {t.token.trim() || '∅'}
                  {t.entity_type && (
                    <span className="ml-1 opacity-60">{t.entity_type}</span>
                  )}
                </span>
              ))}
          </div>
        </div>
      )}
    </div>
  )
}
