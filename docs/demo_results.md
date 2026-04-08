# System Demo Evidence — HalluScan
**Author:** Chalani Dinitha (20211032)  
**Date:** Day 24 — Week 5  
**Branch:** deploy/production-config

---

## Demo Evidence: 5 Prompts Tested

This document records the system's behaviour on 5 key demo prompts,
serving as written evidence of the system working end-to-end.

---

### Demo 1 — Geography Misconception

**Prompt:** `"The capital of Australia is Sydney, which"`  
**Threshold:** 0.35  
**Max tokens:** 20

**System output:**
```
Generated: " is a city in New South Wales, Australia"
EAT tokens detected: ['Sydney', 'New South Wales', 'Australia']
Overall risk: 0.43 (Suspicious)
Flagged tokens: Sydney (score: 0.41)
Processing time: ~2000ms
```

**Research evidence:**
- ✅ Gap 2: Only "Sydney" flagged, not the whole sentence
- ✅ FR9: Exact token identified
- ✅ FR11: Warning banner triggered
- ✅ NFR1: Under 5 seconds

---

### Demo 2 — Historical Date

**Prompt:** `"Napoleon Bonaparte was born in the year"`  
**Threshold:** 0.35  
**Max tokens:** 15

**System output:**
```
Generated: " 1769 in Ajaccio, Corsica"
EAT tokens detected: ['1769', 'Corsica', 'Napoleon Bonaparte']
Overall risk: 0.38 (Suspicious)
Flagged tokens: ['1769'] (score: 0.37)
Processing time: ~1800ms
```

**Research evidence:**
- ✅ Date token correctly identified as EAT
- ✅ Person name token identified as EAT
- ✅ NFR4: Identical result on repeat run

---

### Demo 3 — Scientific Myth

**Prompt:** `"Humans use only 10 percent of their brain, which means"`  
**Threshold:** 0.35  
**Max tokens:** 20

**System output:**
```
Generated: " that we have a lot of room to grow"
EAT tokens detected: ['10']
Overall risk: 0.36 (Suspicious)
Flagged tokens: ['10'] (score: 0.36)
Processing time: ~2100ms
```

**Research evidence:**
- ✅ Percentage number identified as EAT
- ✅ Common misconception triggers detection
- ✅ Gap 2: Only "10" flagged, not the word "percent" or rest

---

### Demo 4 — Nobel Prize Year

**Prompt:** `"Albert Einstein won the Nobel Prize in Physics in"`  
**Threshold:** 0.35  
**Max tokens:** 15

**System output:**
```
Generated: " 1921. He was awarded the prize for"
EAT tokens detected: ['1921', 'Einstein', 'Nobel', 'Physics']
Overall risk: 0.41 (Suspicious)
Flagged tokens: ['1921'] (score: 0.39)
Processing time: ~1600ms
```

**Research evidence:**
- ✅ Year correctly identified as highest-risk EAT
- ✅ Multiple EATs found but only uncertain one flagged
- ✅ FR12: Per-token confidence score displayed

---

### Demo 5 — Great Wall Date

**Prompt:** `"The Great Wall of China was built in"`  
**Threshold:** 0.35  
**Max tokens:** 20

**System output:**
```
Generated: " the 7th century BC by Emperor Qin Shi Huang"
EAT tokens detected: ['China', '7th', 'Qin Shi Huang']
Overall risk: 0.45 (Suspicious)
Flagged tokens: ['7th', 'Qin Shi Huang'] (score: 0.42, 0.46)
Processing time: ~2200ms
```

**Research evidence:**
- ✅ Multiple EATs detected in single response
- ✅ Different scores for different tokens shown
- ✅ Config tab: threshold change instantly affects results

---

## System Performance Summary

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Avg processing time | ~2000ms | < 5000ms | ✅ NFR1 |
| Identical runs | Same output | Reproducible | ✅ NFR4 |
| EAT-only flagging | Confirmed | Gap 2 | ✅ |
| Warning banner | Triggered | FR11 | ✅ |
| Config update | Working | FR14 | ✅ |
| API docs | Available | NFR | ✅ |

---

## How to Record the Demo Video

1. Start both servers (backend + frontend)
2. Open `http://localhost:3000`
3. Use QuickTime Player (Mac) → File → New Screen Recording
4. Follow `docs/demo_script.py` step by step
5. Save as `docs/demo.mp4`

**Recommended screen recording tools on Mac:**
- QuickTime Player (free, built-in)
- Loom (free, uploads to cloud)
- OBS Studio (free, professional)
