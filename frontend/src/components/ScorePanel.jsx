/**
 * ScorePanel.jsx — Score Breakdown Panel
 * Shows overall risk, per-component weights, and token table.
 *
 * Author: Chalani Dinitha (20211032)
 */

export default function ScorePanel({ result }) {
  if (!result) return null

  const risk = result.overall_risk
  const riskColor = risk >= 0.65 ? 'text-red-400' :
                    risk >= 0.45 ? 'text-yellow-400' : 'text-green-400'
  const riskLabel = risk >= 0.65 ? 'HIGH RISK' :
                    risk >= 0.45 ? 'SUSPICIOUS' : 'LOW RISK'
  const barColor  = risk >= 0.65 ? 'bg-red-500' :
                    risk >= 0.45 ? 'bg-yellow-500' : 'bg-green-500'

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 space-y-4">
      <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">
        Score Analysis
      </h2>

      {/* Overall risk bar */}
      <div className="space-y-1.5">
        <div className="flex justify-between text-xs font-mono">
          <span className="text-slate-400">Overall Risk</span>
          <span className={riskColor}>
            {risk.toFixed(3)} · {riskLabel}
          </span>
        </div>
        <div className="h-2 rounded-full bg-slate-800 overflow-hidden">
          <div
            className={`h-full rounded-full transition-all ${barColor}`}
            style={{ width: `${Math.min(risk * 100, 100)}%` }}
          />
        </div>
      </div>

      {/* Scoring formula */}
      <div className="rounded-lg bg-slate-800/60 px-3 py-2 font-mono text-xs
                      text-slate-400 leading-relaxed">
        score = <span className="text-brand-400">0.4</span>×entropy +{' '}
        <span className="text-brand-400">0.4</span>×wasserstein +{' '}
        <span className="text-brand-400">0.2</span>×tsv
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-3 gap-3">
        {[
          { label: 'Flagged', value: result.num_flagged, color: 'text-red-400' },
          { label: 'EAT Tokens', value: result.num_eat_tokens, color: 'text-brand-400' },
          { label: 'Total Tokens', value: result.annotated_tokens.length, color: 'text-slate-300' },
        ].map(({ label, value, color }) => (
          <div key={label} className="rounded-lg bg-slate-800/60 p-3 text-center">
            <p className={`text-xl font-semibold font-mono ${color}`}>{value}</p>
            <p className="text-xs text-slate-500 mt-0.5">{label}</p>
          </div>
        ))}
      </div>

      {/* Token score table — only EAT tokens */}
      {result.num_eat_tokens > 0 && (
        <div>
          <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
            EAT Token Scores
          </p>
          <div className="rounded-lg overflow-hidden border border-slate-800">
            <table className="w-full text-xs font-mono">
              <thead>
                <tr className="bg-slate-800/80 text-slate-500">
                  <th className="text-left px-3 py-2">Token</th>
                  <th className="text-left px-3 py-2">Type</th>
                  <th className="text-right px-3 py-2">Score</th>
                  <th className="text-right px-3 py-2">Risk</th>
                </tr>
              </thead>
              <tbody>
                {result.annotated_tokens
                  .filter(t => t.is_eat)
                  .map((t, i) => (
                    <tr
                      key={i}
                      className={`border-t border-slate-800 ${
                        t.is_flagged ? 'bg-red-500/5' :
                        t.risk_level === 'suspicious' ? 'bg-yellow-500/5' : ''
                      }`}
                    >
                      <td className="px-3 py-2 text-slate-300">
                        {t.token.trim() || '(empty)'}
                      </td>
                      <td className="px-3 py-2 text-slate-500">
                        {t.entity_type || '—'}
                      </td>
                      <td className={`px-3 py-2 text-right ${
                        t.hallucination_score >= 0.65 ? 'text-red-400' :
                        t.hallucination_score >= 0.45 ? 'text-yellow-400' :
                        'text-green-400'
                      }`}>
                        {t.hallucination_score.toFixed(4)}
                      </td>
                      <td className="px-3 py-2 text-right text-slate-500">
                        {t.risk_level}
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
