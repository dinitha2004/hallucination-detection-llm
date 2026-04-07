/**
 * App.jsx — Root Application Component
 * Two-panel layout:
 *   LEFT  → QueryInput (user enters prompt)
 *   RIGHT → Results (TokenDisplay, ScorePanel, WarningBanner)
 *
 * Author: Chalani Dinitha (20211032)
 */

import { useState, useEffect } from 'react'
import { detectHallucination, getHealth } from './api/client'
import QueryInput    from './components/QueryInput'
import TokenDisplay  from './components/TokenDisplay'
import ScorePanel    from './components/ScorePanel'
import WarningBanner from './components/WarningBanner'
import { Brain, Wifi, WifiOff } from 'lucide-react'

export default function App() {
  const [result,   setResult]   = useState(null)
  const [loading,  setLoading]  = useState(false)
  const [error,    setError]    = useState(null)
  const [ready,    setReady]    = useState(false)
  const [checking, setChecking] = useState(true)

  /* ── Check backend health on mount ── */
  useEffect(() => {
    const check = async () => {
      try {
        const h = await getHealth()
        setReady(h.pipeline_ready)
      } catch {
        setReady(false)
      } finally {
        setChecking(false)
      }
    }
    check()
  }, [])

  /* ── Submit a prompt ── */
  const handleSubmit = async (prompt, maxTokens) => {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const data = await detectHallucination(prompt, maxTokens)
      setResult(data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Detection failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex flex-col">

      {/* ── Header ── */}
      <header className="border-b border-slate-800 px-6 py-4 flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-brand-500 flex items-center justify-center">
          <Brain size={18} className="text-white" />
        </div>
        <div>
          <h1 className="text-lg font-semibold tracking-tight">HalluScan</h1>
          <p className="text-xs text-slate-500 font-mono">
            Fine-Grained Hallucination Detection · Chalani Dinitha
          </p>
        </div>

        {/* Backend status pill */}
        <div className="ml-auto">
          {checking ? (
            <span className="text-xs text-slate-500 font-mono animate-pulse">
              connecting…
            </span>
          ) : ready ? (
            <span className="flex items-center gap-1.5 text-xs text-safe-400 font-mono">
              <Wifi size={12} /> API ready
            </span>
          ) : (
            <span className="flex items-center gap-1.5 text-xs text-danger-400 font-mono">
              <WifiOff size={12} /> API offline
            </span>
          )}
        </div>
      </header>

      {/* ── Main two-panel layout ── */}
      <div className="flex flex-1 overflow-hidden">

        {/* LEFT PANEL — Query Input */}
        <aside className="w-96 border-r border-slate-800 flex flex-col p-6 gap-6 overflow-y-auto scrollbar-thin">
          <QueryInput
            onSubmit={handleSubmit}
            loading={loading}
            disabled={!ready && !checking}
          />

          {/* Config reminder */}
          <div className="rounded-xl border border-slate-800 p-4 bg-slate-900/50">
            <p className="text-xs text-slate-500 font-mono leading-relaxed">
              Threshold: 0.65 · Suspicious: 0.45<br/>
              Weights: 0.4 entropy · 0.4 wass · 0.2 tsv<br/>
              Layers: [18, 20, 22]
            </p>
          </div>

          {/* Legend */}
          <div className="space-y-2">
            <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
              Risk Legend
            </p>
            <div className="space-y-1.5 text-sm font-mono">
              <div className="flex items-center gap-2">
                <span className="token-hallucinated text-xs">TOKEN</span>
                <span className="text-slate-400 text-xs">Hallucinated (score ≥ 0.65)</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="token-suspicious text-xs">TOKEN</span>
                <span className="text-slate-400 text-xs">Suspicious (0.45–0.65)</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-slate-300 text-xs border border-slate-700 px-0.5 rounded">
                  TOKEN
                </span>
                <span className="text-slate-400 text-xs">Safe (score &lt; 0.45)</span>
              </div>
            </div>
          </div>
        </aside>

        {/* RIGHT PANEL — Results */}
        <main className="flex-1 flex flex-col overflow-y-auto scrollbar-thin p-6 gap-6">

          {/* Not ready state */}
          {!checking && !ready && (
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center space-y-3">
                <WifiOff size={40} className="text-slate-600 mx-auto" />
                <p className="text-slate-400 font-mono text-sm">
                  Backend not reachable
                </p>
                <p className="text-slate-600 text-xs font-mono">
                  Run: uvicorn backend.main:app --reload --port 8000
                </p>
              </div>
            </div>
          )}

          {/* Empty state */}
          {ready && !loading && !result && !error && (
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center space-y-3 max-w-sm">
                <Brain size={48} className="text-slate-700 mx-auto" />
                <p className="text-slate-400 font-mono text-sm">
                  Enter a prompt to analyse hallucination signals
                </p>
                <p className="text-slate-600 text-xs font-mono">
                  Try: "The capital of France is" or<br/>
                  "Einstein was born in 1879 in Germany"
                </p>
              </div>
            </div>
          )}

          {/* Loading state */}
          {loading && (
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center space-y-4">
                <div className="w-10 h-10 border-2 border-brand-500 border-t-transparent
                                rounded-full animate-spin mx-auto" />
                <p className="text-slate-400 font-mono text-sm animate-pulse">
                  Analysing hidden states…
                </p>
                <p className="text-slate-600 text-xs font-mono">
                  Running pipeline: EAT → Hidden States → Shift → Score
                </p>
              </div>
            </div>
          )}

          {/* Error state */}
          {error && (
            <div className="rounded-xl border border-danger-500/30 bg-danger-500/10 p-4">
              <p className="text-danger-400 font-mono text-sm">{error}</p>
            </div>
          )}

          {/* Results */}
          {result && !loading && (
            <>
              {result.hallucination_detected && (
                <WarningBanner result={result} />
              )}
              <TokenDisplay result={result} />
              <ScorePanel result={result} />
            </>
          )}
        </main>
      </div>
    </div>
  )
}
