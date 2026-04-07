export default function WarningBanner({ result }) {
  if (!result?.hallucination_detected) return null
  const flagged = result.annotated_tokens.filter(t=>t.is_flagged).map(t=>t.token.trim()).filter(Boolean)
  return (
    <div style={{background:'rgba(239,68,68,0.1)',border:'1px solid rgba(239,68,68,0.3)',borderRadius:12,padding:16,display:'flex',alignItems:'flex-start',gap:12}}>
      <span style={{fontSize:18,flexShrink:0}}>⚠️</span>
      <div>
        <div style={{fontWeight:600,color:'#fca5a5',fontSize:14}}>Hallucination detected — {result.num_flagged} token{result.num_flagged!==1?'s':''} flagged</div>
        {flagged.length>0 && <div style={{fontSize:12,fontFamily:'monospace',color:'rgba(252,165,165,0.8)',marginTop:4}}>Flagged: {flagged.map(t=>`"${t}"`).join(', ')}</div>}
        <div style={{fontSize:12,color:'rgba(252,165,165,0.6)',marginTop:4}}>Risk score: {result.overall_risk.toFixed(3)}</div>
      </div>
    </div>
  )
}
