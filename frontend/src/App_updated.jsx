/**
 * App.jsx — Root Application with ConfigPanel + ExperimentLog
 * Author: Chalani Dinitha (20211032)
 */
import { useState, useEffect } from 'react'
import { detectHallucination, getHealth } from './api/client'
import QueryInput    from './components/QueryInput'
import TokenDisplay  from './components/TokenDisplay'
import ScorePanel    from './components/ScorePanel'
import WarningBanner from './components/WarningBanner'
import ConfigPanel   from './components/ConfigPanel'
import ExperimentLog from './components/ExperimentLog'

export default function App() {
  const [result,  setResult]  = useState(null)
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState(null)
  const [ready,   setReady]   = useState(false)
  const [history, setHistory] = useState([])
  const [tab,     setTab]     = useState('results') // 'results' | 'config' | 'log'

  useEffect(() => {
    getHealth()
      .then(h => setReady(h.pipeline_ready))
      .catch(() => setReady(false))
  }, [])

  const handleSubmit = async (prompt, maxTokens) => {
    setLoading(true); setError(null); setResult(null)
    try {
      const data = await detectHallucination(prompt, maxTokens)
      setResult(data)
      // Add to experiment log
      setHistory(prev => [...prev, { prompt, ...data }])
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Detection failed. Is the backend running?')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{display:'flex',flexDirection:'column',minHeight:'100vh',background:'#020617',color:'#f1f5f9'}}>

      {/* ── Header ── */}
      <header style={{padding:'14px 24px',borderBottom:'1px solid #1e293b',display:'flex',alignItems:'center',gap:12,flexShrink:0}}>
        <div style={{width:32,height:32,background:'#4f6ef7',borderRadius:8,display:'flex',alignItems:'center',justifyContent:'center',fontWeight:'bold',fontSize:15,color:'white'}}>H</div>
        <div>
          <div style={{fontWeight:600,fontSize:16,fontFamily:"'DM Sans',sans-serif"}}>HalluScan</div>
          <div style={{fontSize:11,color:'#64748b',fontFamily:'monospace'}}>Fine-Grained Hallucination Detection · Chalani Dinitha (20211032)</div>
        </div>
        <div style={{marginLeft:'auto',display:'flex',alignItems:'center',gap:16}}>
          {/* Tab buttons */}
          {['results','config','log'].map(t => (
            <button key={t} onClick={() => setTab(t)}
              style={{background:'none',border:'none',cursor:'pointer',fontSize:12,fontFamily:'monospace',
                color: tab===t ? '#4f6ef7' : '#475569',
                fontWeight: tab===t ? 600 : 400,
                borderBottom: tab===t ? '2px solid #4f6ef7' : '2px solid transparent',
                paddingBottom:2, textTransform:'capitalize'}}>
              {t === 'log' ? `Log (${history.length})` : t}
            </button>
          ))}
          <div style={{fontSize:12,fontFamily:'monospace',color:ready?'#34d399':'#f87171',marginLeft:8}}>
            {ready ? '● API ready' : '● API offline'}
          </div>
        </div>
      </header>

      {/* ── Main ── */}
      <div style={{display:'flex',flex:1,overflow:'hidden'}}>

        {/* LEFT PANEL */}
        <aside style={{width:360,borderRight:'1px solid #1e293b',padding:20,overflowY:'auto',display:'flex',flexDirection:'column',gap:16,flexShrink:0}}>
          <QueryInput onSubmit={handleSubmit} loading={loading} disabled={!ready} />

          {/* Legend */}
          <div style={{background:'#0f172a',border:'1px solid #1e293b',borderRadius:12,padding:14}}>
            <div style={{fontSize:11,fontWeight:600,color:'#475569',textTransform:'uppercase',letterSpacing:1,marginBottom:10}}>Risk Legend</div>
            <div style={{display:'flex',flexDirection:'column',gap:7,fontSize:12,fontFamily:'monospace'}}>
              <div style={{display:'flex',alignItems:'center',gap:8}}>
                <span style={{background:'rgba(239,68,68,0.2)',color:'#fca5a5',padding:'1px 8px',borderRadius:4,outline:'1px solid rgba(239,68,68,0.4)',fontWeight:600}}>TOKEN</span>
                <span style={{color:'#64748b'}}>Hallucinated ≥ 0.65</span>
              </div>
              <div style={{display:'flex',alignItems:'center',gap:8}}>
                <span style={{background:'rgba(250,204,21,0.2)',color:'#fde047',padding:'1px 8px',borderRadius:4}}>TOKEN</span>
                <span style={{color:'#64748b'}}>Suspicious 0.45–0.65</span>
              </div>
              <div style={{display:'flex',alignItems:'center',gap:8}}>
                <span style={{border:'1px solid #334155',padding:'1px 8px',borderRadius:4,color:'#94a3b8'}}>TOKEN</span>
                <span style={{color:'#64748b'}}>Safe &lt; 0.45</span>
              </div>
            </div>
          </div>

          <div style={{fontSize:11,color:'#334155',fontFamily:'monospace',lineHeight:1.7,background:'#0f172a',border:'1px solid #1e293b',borderRadius:12,padding:14}}>
            Score = 0.4×entropy + 0.4×wasserstein + 0.2×tsv<br/>
            Threshold: 0.65 · Layers: [18, 20, 22]<br/>
            Model: facebook/opt-1.3b
          </div>
        </aside>

        {/* RIGHT PANEL */}
        <main style={{flex:1,padding:24,overflowY:'auto',display:'flex',flexDirection:'column',gap:20}}>

          {/* Results Tab */}
          {tab === 'results' && <>
            {!ready && (
              <div style={{flex:1,display:'flex',alignItems:'center',justifyContent:'center',flexDirection:'column',gap:12,color:'#475569',fontFamily:'monospace',fontSize:14}}>
                <div style={{fontSize:40}}>⚠</div>
                <div>Backend not reachable</div>
                <div style={{fontSize:11,color:'#334155'}}>uvicorn backend.main:app --reload --port 8000</div>
              </div>
            )}

            {ready && !loading && !result && !error && (
              <div style={{flex:1,display:'flex',alignItems:'center',justifyContent:'center',flexDirection:'column',gap:12,color:'#475569',fontFamily:'monospace',fontSize:14,textAlign:'center'}}>
                <div style={{fontSize:52}}>🧠</div>
                <div>Enter a prompt to analyse hallucination signals</div>
                <div style={{fontSize:11,color:'#334155'}}>Try: "The capital of France is" or<br/>"Einstein was born in 1879 in Germany"</div>
              </div>
            )}

            {/* Loading spinner */}
            {loading && (
              <div style={{flex:1,display:'flex',alignItems:'center',justifyContent:'center',flexDirection:'column',gap:16}}>
                <div style={{width:44,height:44,border:'3px solid #4f6ef7',borderTopColor:'transparent',borderRadius:'50%',animation:'spin 0.8s linear infinite'}} />
                <div style={{color:'#64748b',fontFamily:'monospace',fontSize:14}}>Analysing hidden states…</div>
                <div style={{color:'#334155',fontFamily:'monospace',fontSize:11}}>Pipeline: EAT → Hidden States → HalluShift → Score → Span Map</div>
                <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>
              </div>
            )}

            {/* Error */}
            {error && (
              <div style={{background:'rgba(239,68,68,0.1)',border:'1px solid rgba(239,68,68,0.3)',borderRadius:12,padding:16,color:'#f87171',fontFamily:'monospace',fontSize:13,lineHeight:1.6}}>
                <div style={{fontWeight:600,marginBottom:4}}>⚠ Detection Error</div>
                {error}
              </div>
            )}

            {/* Results */}
            {result && !loading && (
              <>
                {result.hallucination_detected && <WarningBanner result={result} />}
                <TokenDisplay result={result} />
                <ScorePanel result={result} />
              </>
            )}
          </>}

          {/* Config Tab */}
          {tab === 'config' && (
            <div style={{maxWidth:600}}>
              <ConfigPanel />
            </div>
          )}

          {/* Log Tab */}
          {tab === 'log' && (
            history.length === 0
              ? <div style={{flex:1,display:'flex',alignItems:'center',justifyContent:'center',color:'#475569',fontFamily:'monospace',fontSize:14}}>No queries yet</div>
              : <ExperimentLog history={history} />
          )}
        </main>
      </div>
    </div>
  )
}
