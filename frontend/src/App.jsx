import { useState, useEffect } from 'react'
import { detectHallucination, getHealth } from './api/client'
import QueryInput from './components/QueryInput'
import TokenDisplay from './components/TokenDisplay'
import ScorePanel from './components/ScorePanel'
import WarningBanner from './components/WarningBanner'

export default function App() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [ready, setReady] = useState(false)

  useEffect(() => {
    getHealth().then(h => setReady(h.pipeline_ready)).catch(() => setReady(false))
  }, [])

  const handleSubmit = async (prompt, maxTokens) => {
    setLoading(true); setError(null); setResult(null)
    try {
      setResult(await detectHallucination(prompt, maxTokens))
    } catch(err) {
      setError(err.message || 'Detection failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{display:'flex',flexDirection:'column',minHeight:'100vh'}}>
      <header style={{padding:'16px 24px',borderBottom:'1px solid #1e293b',display:'flex',alignItems:'center',gap:'12px'}}>
        <div style={{width:32,height:32,background:'#4f6ef7',borderRadius:8,display:'flex',alignItems:'center',justifyContent:'center',fontWeight:'bold',fontSize:14}}>H</div>
        <div>
          <div style={{fontWeight:600,fontSize:16}}>HalluScan</div>
          <div style={{fontSize:11,color:'#64748b',fontFamily:'monospace'}}>Fine-Grained Hallucination Detection · Chalani Dinitha</div>
        </div>
        <div style={{marginLeft:'auto',fontSize:12,fontFamily:'monospace',color:ready?'#34d399':'#f87171'}}>
          {ready ? '● API ready' : '● API offline'}
        </div>
      </header>
      <div style={{display:'flex',flex:1,overflow:'hidden'}}>
        <aside style={{width:380,borderRight:'1px solid #1e293b',padding:24,overflowY:'auto',display:'flex',flexDirection:'column',gap:20}}>
          <QueryInput onSubmit={handleSubmit} loading={loading} disabled={!ready} />
          <div style={{background:'#0f172a',border:'1px solid #1e293b',borderRadius:12,padding:16,fontSize:12,fontFamily:'monospace',color:'#475569',lineHeight:1.8}}>
            Threshold: 0.65 · Suspicious: 0.45<br/>
            Weights: 0.4×entropy + 0.4×wass + 0.2×tsv<br/>
            Layers: [18, 20, 22] · Model: OPT-1.3B
          </div>
          <div>
            <div style={{fontSize:11,fontWeight:600,color:'#64748b',textTransform:'uppercase',letterSpacing:1,marginBottom:8}}>Legend</div>
            <div style={{display:'flex',flexDirection:'column',gap:6,fontSize:12,fontFamily:'monospace'}}>
              <div style={{display:'flex',alignItems:'center',gap:8}}><span style={{background:'rgba(239,68,68,0.2)',color:'#fca5a5',padding:'1px 6px',borderRadius:4,outline:'1px solid rgba(239,68,68,0.4)'}}>TOKEN</span><span style={{color:'#64748b'}}>Hallucinated (≥0.65)</span></div>
              <div style={{display:'flex',alignItems:'center',gap:8}}><span style={{background:'rgba(250,204,21,0.2)',color:'#fde047',padding:'1px 6px',borderRadius:4}}>TOKEN</span><span style={{color:'#64748b'}}>Suspicious (0.45–0.65)</span></div>
              <div style={{display:'flex',alignItems:'center',gap:8}}><span style={{border:'1px solid #334155',padding:'1px 6px',borderRadius:4,color:'#cbd5e1'}}>TOKEN</span><span style={{color:'#64748b'}}>Safe (&lt;0.45)</span></div>
            </div>
          </div>
        </aside>
        <main style={{flex:1,padding:24,overflowY:'auto',display:'flex',flexDirection:'column',gap:20}}>
          {!ready && <div style={{flex:1,display:'flex',alignItems:'center',justifyContent:'center',flexDirection:'column',gap:12,color:'#475569',fontFamily:'monospace',fontSize:14}}>
            <div style={{fontSize:32}}>⚠</div>
            <div>Backend not reachable</div>
            <div style={{fontSize:11,color:'#334155'}}>Run: uvicorn backend.main:app --reload --port 8000</div>
          </div>}
          {ready && !loading && !result && !error && <div style={{flex:1,display:'flex',alignItems:'center',justifyContent:'center',flexDirection:'column',gap:12,color:'#475569',fontFamily:'monospace',fontSize:14,textAlign:'center'}}>
            <div style={{fontSize:48}}>🧠</div>
            <div>Enter a prompt to analyse hallucination signals</div>
            <div style={{fontSize:11,color:'#334155'}}>Try: "The capital of France is"</div>
          </div>}
          {loading && <div style={{flex:1,display:'flex',alignItems:'center',justifyContent:'center',flexDirection:'column',gap:16}}>
            <div style={{width:40,height:40,border:'3px solid #4f6ef7',borderTopColor:'transparent',borderRadius:'50%',animation:'spin 1s linear infinite'}} />
            <div style={{color:'#64748b',fontFamily:'monospace',fontSize:14}}>Analysing hidden states…</div>
            <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>
          </div>}
          {error && <div style={{background:'rgba(239,68,68,0.1)',border:'1px solid rgba(239,68,68,0.3)',borderRadius:12,padding:16,color:'#f87171',fontFamily:'monospace',fontSize:13}}>{error}</div>}
          {result && !loading && <>
            {result.hallucination_detected && <WarningBanner result={result} />}
            <TokenDisplay result={result} />
            <ScorePanel result={result} />
          </>}
        </main>
      </div>
    </div>
  )
}
