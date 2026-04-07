/**
 * WarningBanner.jsx — Hallucination Warning Banner
 * Shown at the top of results when hallucination is detected.
 *
 * Author: Chalani Dinitha (20211032)
 */

import { AlertTriangle } from 'lucide-react'

export default function WarningBanner({ result }) {
  if (!result?.hallucination_detected) return null

  const flagged = result.annotated_tokens.filter(t => t.is_flagged)
  const tokens  = flagged.map(t => t.token.trim()).filter(Boolean)

  return (
    <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-4
                    flex items-start gap-3">
      <AlertTriangle size={18} className="text-red-400 flex-shrink-0 mt-0.5" />
      <div className="space-y-1">
        <p className="text-sm font-semibold text-red-300">
          Hallucination detected — {result.num_flagged} token
          {result.num_flagged !== 1 ? 's' : ''} flagged
        </p>
        {tokens.length > 0 && (
          <p className="text-xs font-mono text-red-400/80">
            Flagged: {tokens.map(t => `"${t}"`).join(', ')}
          </p>
        )}
        <p className="text-xs text-red-400/60">
          Overall risk score: {result.overall_risk.toFixed(3)}
        </p>
      </div>
    </div>
  )
}
