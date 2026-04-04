# 🔍 Hallucination Detection System
### Fine-Grained Hallucination Detection Using Hidden States of Large Language Models

> **BEng (Hons) Software Engineering Dissertation**  
> Chalani Dinitha — W1953891 / 20211032  
> IIT (Informatics Institute of Technology) in Collaboration with University of Westminster, UK  
> Supervised by Mr. Saadh Jawwadh

---

## 📋 Project Overview

Large Language Models (LLMs) like GPT, LLaMA, and Claude sometimes generate answers that **sound correct but are factually wrong** — this is called **hallucination**. This system solves two critical gaps in existing hallucination detection research:

| Gap | Problem | This System's Solution |
|-----|---------|------------------------|
| **Gap 1** | Most systems detect hallucinations *after* the full answer is generated | Monitor LLM **hidden states during generation** (TBG Probing + HalluShift) |
| **Gap 2** | Systems mark the *entire* paragraph as wrong, even when only one word is incorrect | **Token-level localization** — highlights only the exact wrong word (EAT + Span Mapper) |

---

## 🏗️ System Architecture

The system runs 4 synchronized modules during LLM text generation:

```
User Prompt
     │
     ▼
┌─────────────────────────────────────────────┐
│  MODULE A: EAT Detection                    │
│  • Identifies Exact Answer Tokens           │
│  • Names, dates, years, numbers, locations  │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│  MODULE B: Hidden State Extraction          │
│  • TBG (Token Before Generating) probing    │
│  • INSIDE Feature Clipping (overconfidence) │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│  MODULE C: Distribution Shift Analysis      │
│  • HalluShift: Wasserstein + Cosine         │
│  • TSV: Truthfulness Separator Vector       │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│  MODULE D: Scoring & Localization           │
│  • Semantic Entropy calculation             │
│  • Span-Level Mapper (EAT + scores)         │
│  • Token-level highlighted output           │
└─────────────────────────────────────────────┘
                      │
                      ▼
     🔴 Highlighted wrong tokens in React UI
```

---

## 🛠️ Technology Stack

| Layer | Technology |
|-------|-----------|
| LLM | LLaMA-3.2-3B-Instruct (fallback: OPT-1.3B) |
| Backend | FastAPI + Python 3.11 |
| Hidden State Extraction | PyTorch `register_forward_hook` |
| Signal Processing | scipy, numpy, scikit-learn |
| EAT Detection | spaCy NER |
| Frontend | React + TailwindCSS |
| Evaluation | TruthfulQA + TriviaQA datasets |
| Experiment Tracking | MLflow |
| Testing | pytest |
| Containerization | Docker + docker-compose |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 20+
- Git
- 8GB RAM minimum (16GB recommended for LLM)

### 1. Clone the repository
```bash
git clone https://github.com/dinitha2004/hallucination-detection-llm.git
cd hallucination-detection-llm
```

### 2. Set up Python environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate       # Mac/Linux
# OR
venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt

# Download spaCy English model
python -m spacy download en_core_web_sm
```

### 3. Configure environment
```bash
# Copy example config
cp .env.example .env

# Edit .env with your HuggingFace token and settings
# (Get token from https://huggingface.co/settings/tokens)
```

### 4. Start the backend
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Start the frontend
```bash
cd frontend
npm install
npm start
```

### 6. Open the app
Visit `http://localhost:3000` — type a question and see hallucinations highlighted!

---

## 🐳 Docker (Easiest Way)
```bash
docker-compose up --build
```
- Frontend: http://localhost
- Backend API: http://localhost:8000/docs

---

## 📊 Evaluation

Run evaluation on TruthfulQA and TriviaQA benchmarks:
```bash
python -m backend.evaluation.experiment_runner
```

View results in MLflow UI:
```bash
mlflow ui
# Visit http://localhost:5000
```

---

## 🧪 Testing
```bash
# Run all tests
pytest backend/tests/ -v

# Run specific module test
pytest backend/tests/test_module_b.py -v
```

---

## 📁 Project Structure
```
hallucination-detection-llm/
├── backend/
│   ├── config.py              # All configuration (thresholds, layers, weights)
│   ├── main.py                # FastAPI app + endpoints
│   ├── schemas.py             # API request/response models
│   ├── llm/
│   │   ├── model_loader.py    # LLM loading with hidden state output
│   │   └── inference_engine.py # Token generation + state capture
│   ├── modules/
│   │   ├── module_a_eat.py    # EAT detection (Gap 2)
│   │   ├── module_b_hidden.py # Hidden state extraction (Gap 1)
│   │   ├── module_c_hallushift.py  # Distribution shift + TSV
│   │   └── module_d_scoring.py    # Entropy + Span mapper
│   ├── pipeline/
│   │   └── detection_pipeline.py  # Orchestrates all 4 modules
│   └── evaluation/
│       ├── dataset_loader.py  # TruthfulQA + TriviaQA
│       ├── metrics.py         # F1, precision, recall
│       └── experiment_runner.py  # MLflow logging
├── frontend/
│   └── src/
│       ├── App.jsx
│       └── components/
│           ├── QueryInput.jsx    # User prompt input
│           ├── TokenDisplay.jsx  # Highlighted token output
│           ├── ScorePanel.jsx    # Confidence scores
│           └── WarningBanner.jsx # Hallucination warning
├── data/                      # Datasets and memory bank
├── experiments/               # Results, notebooks, plots
└── docs/                      # Architecture docs, API reference
```

---

## 🔬 Research Details

- **Dataset**: TruthfulQA (primary), TriviaQA (secondary)
- **Evaluation Metrics**: Token-level F1, Span Precision/Recall, Latency
- **Baseline Comparison**: Post-hoc detection methods from literature
- **Ablation Study**: 4 conditions (Full / no TSV / no EAT / no feature clipping)

---

## 📧 Contact

**Chalani Dinitha**  
Email: chalani.20211032@iit.ac.lk  
GitHub: [@dinitha2004](https://github.com/dinitha2004)

---

*Submitted in partial fulfillment of the requirements for the BEng (Hons) Software Engineering degree at the University of Westminster.*
