import { useState } from 'react'
function Token({ t }) {
  const [show, setShow] = useState(false)
  const style = t.is_flagged
    ? {background:'rgba(239,68,68,0.2)',color:'#fca5a5',borderRadius:4,padding:'0 2px',fontWeight:600,outline:'1px solid rgba(239,68,68,0.4)',cursor:'help'}
    : t.risk_level==='suspicious' && t.is_eat
    ? {background:'rgba(250,204,21,0.2)',color:'#fde047',borderRadius:4,padding:'0 2px',cursor:'help'}
    : {color:'#cbd5e1'}
  return (
    <span style={{position:'relative',display:'inline'}}>
      <span style={{fontFamily:'monospace',fontSize:14,whiteSpace:'pre-wrap',...style}} onMouseEnter={()=>t.is_eat&&setShow(true)} onMouseLeave={()=>setShow(false)}>{t.token}</span>
      {show && <span style={{position:'absolute',bottom:'100%',left:'50%',transform:'translateX(-50%)',background:'#1e293b',border:'1px solid #334155',borderRadius:8,padding:'6px 10px',fontSize:11,fontFamily:'monospace',whiteSpace:'nowrap',zIndex:50,boxShadow:'0 4px 20px rgba(0,0,0,0.5)'}}>
        <span style={{display:'block',color:'#94a3b8'}}>score: <span style={{color:t.hallucination_score>=0.65?'#f87171':t.hallucination_score>=0.45?'#fbbf24':'#34d399',fontWeight:600}}>{t.hallucination_score.toFixed(3)}</span></span>
        {t.is_eat && <span style={{display:'block',color:'#818cf8'}}>EAT · {t.entity_type||'entity'}</span>}
        <span style={{display:'block',color:'#475569'}}>{t.risk_level}</span>
      </span>}
    </span>
  )
}
export default function TokenDisplay({ result }) {
  if (!result) return null
  return (
    <div style={{background:'rgba(15,23,42,0.6)',border:'1px solid #1e293b',borderRadius:16,padding:20}}>
      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:16}}>
        <div style={{fontSize:12,fontWeight:600,color:'#94a3b8',textTransform:'uppercase',letterSpacing:1}}>Generated Response</div>
        <div style={{fontSize:11,fontFamily:'monospace',color:'#475569'}}>{result.annotated_tokens.length} tokens · {result.num_eat_tokens} EATs · {result.processing_time_ms.toFixed(0)}ms</div>
      </div>
      <div style={{lineHeight:2.2}}>{result.annotated_tokens.map((t,i)=><Token key={i} t={t}/>)}</div>
      {result.num_eat_tokens>0 && <div style={{marginTop:16,paddingTop:16,borderTop:'1px solid #1e293b'}}>
        <div style={{fontSize:11,fontFamily:'monospace',color:'#475569',marginBottom:8}}>EAT tokens detected:</div>
        <div style={{display:'flex',flexWrap:'wrap',gap:6}}>
          {result.annotated_tokens.filter(t=>t.is_eat).map((t,i)=>(
            <span key={i} style={{fontSize:11,fontFamily:'monospace',padding:'2px 8px',borderRadius:6,border:`1px solid ${t.is_flagged?'rgba(239,68,68,0.4)':t.risk_level==='suspicious'?'rgba(250,204,21,0.4)':'#334155'}`,background:t.is_flagged?'rgba(239,68,68,0.1)':t.risk_level==='suspicious'?'rgba(250,204,21,0.1)':'#0f172a',color:t.is_flagged?'#f87171':t.risk_level==='suspicious'?'#fbbf24':'#64748b'}}>
              {t.token.trim()||'∅'} {t.entity_type&&<span style={{opacity:0.6}}>{t.entity_type}</span>}
            </span>
          ))}
        </div>
      </div>}
    </div>
  )
}
