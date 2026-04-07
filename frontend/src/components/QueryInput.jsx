import { useState } from 'react'
const EXAMPLES = ['The capital of France is','Einstein was born in 1879 in Germany','Shakespeare wrote Hamlet in approximately','NASA was founded on October 1,']
export default function QueryInput({ onSubmit, loading, disabled }) {
  const [prompt, setPrompt] = useState('')
  const [maxTokens, setMaxTokens] = useState(30)
  const submit = () => { if(prompt.trim() && !loading && !disabled) onSubmit(prompt.trim(), maxTokens) }
  return (
    <div style={{display:'flex',flexDirection:'column',gap:16}}>
      <div>
        <div style={{fontSize:11,fontWeight:600,color:'#64748b',textTransform:'uppercase',letterSpacing:1,marginBottom:8}}>Prompt</div>
        <textarea value={prompt} onChange={e=>setPrompt(e.target.value)} onKeyDown={e=>{if(e.key==='Enter'&&(e.metaKey||e.ctrlKey))submit()}}
          placeholder="Enter a factual question or statement…" rows={5} disabled={disabled||loading}
          style={{width:'100%',background:'#0f172a',border:'1px solid #334155',borderRadius:12,color:'#f1f5f9',fontFamily:'monospace',fontSize:13,padding:12,resize:'none',outline:'none',boxSizing:'border-box',opacity:disabled||loading?0.5:1}} />
        <div style={{fontSize:11,color:'#334155',fontFamily:'monospace',marginTop:4}}>⌘+Enter to submit</div>
      </div>
      <div>
        <div style={{fontSize:11,fontWeight:600,color:'#64748b',textTransform:'uppercase',letterSpacing:1,marginBottom:8}}>
          Max tokens: <span style={{color:'#4f6ef7'}}>{maxTokens}</span>
        </div>
        <input type="range" min={10} max={100} step={5} value={maxTokens} onChange={e=>setMaxTokens(Number(e.target.value))} style={{width:'100%',accentColor:'#4f6ef7'}} disabled={disabled||loading} />
        <div style={{display:'flex',justifyContent:'space-between',fontSize:11,color:'#334155',fontFamily:'monospace',marginTop:2}}><span>10</span><span>100</span></div>
      </div>
      <button onClick={submit} disabled={!prompt.trim()||loading||disabled}
        style={{width:'100%',background:'#4f6ef7',color:'white',border:'none',borderRadius:12,padding:'12px 0',fontWeight:600,fontSize:14,cursor:'pointer',opacity:!prompt.trim()||loading||disabled?0.4:1}}>
        {loading ? '⏳ Analysing…' : '🔍 Detect Hallucination'}
      </button>
      <div>
        <div style={{fontSize:11,fontWeight:600,color:'#475569',textTransform:'uppercase',letterSpacing:1,marginBottom:8}}>Examples</div>
        {EXAMPLES.map(p=>(
          <button key={p} onClick={()=>setPrompt(p)} disabled={loading} style={{width:'100%',background:'none',border:'none',color:'#475569',fontFamily:'monospace',fontSize:11,textAlign:'left',padding:'6px 8px',cursor:'pointer',borderRadius:8,marginBottom:2,display:'block'}}>
            {p}
          </button>
        ))}
      </div>
    </div>
  )
}
