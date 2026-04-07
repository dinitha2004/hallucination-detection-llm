/**
 * ConfigPanel.jsx — Live Configuration Panel (FR14)
 * Sliders for HALLUCINATION_THRESHOLD and SUSPICIOUS_THRESHOLD_LOW.
 * Calls POST /api/config when sliders change.
 * Author: Chalani Dinitha (20211032)
 */
import { useState, useEffect } from 'react'
import { getConfig, updateConfig } from '../api/client'

export default function ConfigPanel() {
  const [config, setConfig]   = useState(null)
  const [saved,  setSaved]    = useState(false)
  const [error,  setError]    = useState(null)
  const [halluThreshold, setHalluThreshold] = useState(0.65)
  const [suspThreshold,  setSuspThreshold]  = useState(0.45)

  // Load current config on mount
  useEffect(() => {
    getConfig()
      .then(c => {
        setConfig(c)
        setHalluThreshold(c.hallucination_threshold)
        setSuspThreshold(c.suspicious_threshold_low)
      })
      .catch(() => setError('Could not load config'))
  }, [])

  const handleSave = async () => {
    try {
      setError(null)
      const res = await updateConfig({
        hallucination_threshold: halluThreshold,
        suspicious_threshold_low: suspThreshold,
      })
      setConfig(res.updated_config)
      setSaved(true)
      setTimeout(() => setSaved(false), 2000)
    } catch {
      setError('Failed to update config')
    }
  }

  return (
    <div style={{background:'rgba(15,23,42,0.6)',border:'1px solid #1e293b',borderRadius:16,padding:20,display:'flex',flexDirection:'column',gap:16}}>
      <div style={{fontSize:12,fontWeight:600,color:'#94a3b8',textTransform:'uppercase',letterSpacing:1}}>
        Configuration (FR14)
      </div>

      {/* Hallucination Threshold */}
      <div>
        <div style={{display:'flex',justifyContent:'space-between',fontSize:12,fontFamily:'monospace',marginBottom:8}}>
          <span style={{color:'#64748b'}}>Hallucination Threshold</span>
          <span style={{color:'#f87171',fontWeight:600}}>{halluThreshold.toFixed(2)}</span>
        </div>
        <input type="range" min={0.1} max={0.9} step={0.05}
          value={halluThreshold}
          onChange={e => setHalluThreshold(parseFloat(e.target.value))}
          style={{width:'100%',accentColor:'#ef4444'}} />
        <div style={{display:'flex',justifyContent:'space-between',fontSize:10,color:'#334155',fontFamily:'monospace',marginTop:2}}>
          <span>0.10 (sensitive)</span><span>0.90 (strict)</span>
        </div>
      </div>

      {/* Suspicious Threshold */}
      <div>
        <div style={{display:'flex',justifyContent:'space-between',fontSize:12,fontFamily:'monospace',marginBottom:8}}>
          <span style={{color:'#64748b'}}>Suspicious Threshold</span>
          <span style={{color:'#fbbf24',fontWeight:600}}>{suspThreshold.toFixed(2)}</span>
        </div>
        <input type="range" min={0.1} max={0.8} step={0.05}
          value={suspThreshold}
          onChange={e => setSuspThreshold(parseFloat(e.target.value))}
          style={{width:'100%',accentColor:'#f59e0b'}} />
        <div style={{display:'flex',justifyContent:'space-between',fontSize:10,color:'#334155',fontFamily:'monospace',marginTop:2}}>
          <span>0.10</span><span>0.80</span>
        </div>
      </div>

      {/* Model info */}
      {config && (
        <div style={{background:'rgba(15,23,42,0.8)',borderRadius:8,padding:'8px 12px',fontSize:11,fontFamily:'monospace',color:'#475569',lineHeight:1.8}}>
          Model: {config.model_name}<br/>
          Layers: [{config.target_layers?.join(', ')}]<br/>
          Weights: {config.weight_entropy}×H + {config.weight_wasserstein}×W + {config.weight_tsv}×TSV
        </div>
      )}

      {/* Save button */}
      <button onClick={handleSave}
        style={{width:'100%',background:saved?'#059669':'#1e293b',color:saved?'white':'#94a3b8',border:'1px solid',borderColor:saved?'#059669':'#334155',borderRadius:10,padding:'10px 0',fontWeight:600,fontSize:13,cursor:'pointer',transition:'all 0.2s',fontFamily:'monospace'}}>
        {saved ? '✓ Saved!' : 'Apply Changes'}
      </button>

      {error && (
        <div style={{fontSize:12,color:'#f87171',fontFamily:'monospace',textAlign:'center'}}>
          {error}
        </div>
      )}
    </div>
  )
}
