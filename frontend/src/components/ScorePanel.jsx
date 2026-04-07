export default function ScorePanel({ result }) {
  if (!result) return null
  const r = result.overall_risk
  const color = r>=0.65?'#ef4444':r>=0.45?'#f59e0b':'#10b981'
  const label = r>=0.65?'HIGH RISK':r>=0.45?'SUSPICIOUS':'LOW RISK'
  return (
    <div style={{background:'rgba(15,23,42,0.6)',border:'1px solid #1e293b',borderRadius:16,padding:20,display:'flex',flexDirection:'column',gap:16}}>
      <div style={{fontSize:12,fontWeight:600,color:'#94a3b8',textTransform:'uppercase',letterSpacing:1}}>Score Analysis</div>
      <div>
        <div style={{display:'flex',justifyContent:'space-between',fontSize:12,fontFamily:'monospace',marginBottom:6}}>
          <span style={{color:'#64748b'}}>Overall Risk</span>
          <span style={{color,fontWeight:600}}>{r.toFixed(3)} · {label}</span>
        </div>
        <div style={{height:8,background:'#1e293b',borderRadius:4,overflow:'hidden'}}>
          <div style={{height:'100%',background:color,width:`${Math.min(r*100,100)}%`,borderRadius:4,transition:'width 0.5s'}} />
        </div>
      </div>
      <div style={{background:'rgba(15,23,42,0.8)',borderRadius:8,padding:'8px 12px',fontFamily:'monospace',fontSize:12,color:'#64748b'}}>
        score = <span style={{color:'#818cf8'}}>0.4</span>×entropy + <span style={{color:'#818cf8'}}>0.4</span>×wasserstein + <span style={{color:'#818cf8'}}>0.2</span>×tsv
      </div>
      <div style={{display:'grid',gridTemplateColumns:'1fr 1fr 1fr',gap:12}}>
        {[['Flagged',result.num_flagged,'#f87171'],['EAT Tokens',result.num_eat_tokens,'#818cf8'],['Total',result.annotated_tokens.length,'#94a3b8']].map(([l,v,c])=>(
          <div key={l} style={{background:'rgba(15,23,42,0.8)',borderRadius:10,padding:12,textAlign:'center'}}>
            <div style={{fontSize:22,fontWeight:700,fontFamily:'monospace',color:c}}>{v}</div>
            <div style={{fontSize:11,color:'#475569',marginTop:2}}>{l}</div>
          </div>
        ))}
      </div>
      {result.num_eat_tokens>0 && <div>
        <div style={{fontSize:11,fontWeight:600,color:'#475569',textTransform:'uppercase',letterSpacing:1,marginBottom:8}}>EAT Token Scores</div>
        <div style={{border:'1px solid #1e293b',borderRadius:10,overflow:'hidden'}}>
          <table style={{width:'100%',fontSize:12,fontFamily:'monospace',borderCollapse:'collapse'}}>
            <thead><tr style={{background:'rgba(30,41,59,0.8)',color:'#475569'}}>
              <th style={{textAlign:'left',padding:'8px 12px'}}>Token</th>
              <th style={{textAlign:'left',padding:'8px 12px'}}>Type</th>
              <th style={{textAlign:'right',padding:'8px 12px'}}>Score</th>
              <th style={{textAlign:'right',padding:'8px 12px'}}>Risk</th>
            </tr></thead>
            <tbody>{result.annotated_tokens.filter(t=>t.is_eat).map((t,i)=>(
              <tr key={i} style={{borderTop:'1px solid #1e293b',background:t.is_flagged?'rgba(239,68,68,0.05)':t.risk_level==='suspicious'?'rgba(245,158,11,0.05)':''}}>
                <td style={{padding:'8px 12px',color:'#cbd5e1'}}>{t.token.trim()||'∅'}</td>
                <td style={{padding:'8px 12px',color:'#475569'}}>{t.entity_type||'—'}</td>
                <td style={{padding:'8px 12px',textAlign:'right',color:t.hallucination_score>=0.65?'#f87171':t.hallucination_score>=0.45?'#fbbf24':'#34d399',fontWeight:600}}>{t.hallucination_score.toFixed(4)}</td>
                <td style={{padding:'8px 12px',textAlign:'right',color:'#475569'}}>{t.risk_level}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      </div>}
    </div>
  )
}
