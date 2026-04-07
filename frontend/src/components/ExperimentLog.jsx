/**
 * ExperimentLog.jsx — Past Queries Table
 * Shows history of all queries with their scores.
 * Author: Chalani Dinitha (20211032)
 */
import { useState } from 'react'

export default function ExperimentLog({ history }) {
  const [expanded, setExpanded] = useState(false)

  if (!history || history.length === 0) return null

  const shown = expanded ? history : history.slice(-5)

  return (
    <div style={{background:'rgba(15,23,42,0.6)',border:'1px solid #1e293b',borderRadius:16,padding:20}}>
      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:16}}>
        <div style={{fontSize:12,fontWeight:600,color:'#94a3b8',textTransform:'uppercase',letterSpacing:1}}>
          Experiment Log
        </div>
        <div style={{display:'flex',alignItems:'center',gap:12}}>
          <span style={{fontSize:11,fontFamily:'monospace',color:'#475569'}}>
            {history.length} queries
          </span>
          {history.length > 5 && (
            <button
              onClick={() => setExpanded(!expanded)}
              style={{fontSize:11,fontFamily:'monospace',color:'#4f6ef7',background:'none',border:'none',cursor:'pointer'}}
            >
              {expanded ? 'Show less' : `Show all ${history.length}`}
            </button>
          )}
        </div>
      </div>

      <div style={{border:'1px solid #1e293b',borderRadius:10,overflow:'hidden'}}>
        <table style={{width:'100%',fontSize:12,fontFamily:'monospace',borderCollapse:'collapse'}}>
          <thead>
            <tr style={{background:'rgba(30,41,59,0.8)',color:'#475569'}}>
              <th style={{textAlign:'left',padding:'8px 12px'}}>#</th>
              <th style={{textAlign:'left',padding:'8px 12px'}}>Prompt</th>
              <th style={{textAlign:'right',padding:'8px 12px'}}>Risk</th>
              <th style={{textAlign:'right',padding:'8px 12px'}}>EATs</th>
              <th style={{textAlign:'right',padding:'8px 12px'}}>Flagged</th>
              <th style={{textAlign:'right',padding:'8px 12px'}}>ms</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((entry, i) => {
              const idx = history.length - shown.length + i + 1
              const risk = entry.overall_risk
              const riskColor = risk >= 0.65 ? '#f87171' : risk >= 0.45 ? '#fbbf24' : '#34d399'
              return (
                <tr key={i} style={{borderTop:'1px solid #1e293b',background:entry.hallucination_detected?'rgba(239,68,68,0.03)':''}}>
                  <td style={{padding:'8px 12px',color:'#475569'}}>{idx}</td>
                  <td style={{padding:'8px 12px',color:'#94a3b8',maxWidth:200,overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap'}}>
                    {entry.prompt}
                  </td>
                  <td style={{padding:'8px 12px',textAlign:'right',color:riskColor,fontWeight:600}}>
                    {risk.toFixed(3)}
                  </td>
                  <td style={{padding:'8px 12px',textAlign:'right',color:'#818cf8'}}>
                    {entry.num_eat_tokens}
                  </td>
                  <td style={{padding:'8px 12px',textAlign:'right',color:entry.num_flagged>0?'#f87171':'#475569'}}>
                    {entry.num_flagged}
                  </td>
                  <td style={{padding:'8px 12px',textAlign:'right',color:'#475569'}}>
                    {entry.processing_time_ms.toFixed(0)}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
