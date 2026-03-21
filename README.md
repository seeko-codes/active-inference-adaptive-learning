# Active Inference Adaptive Learning System

An algebra tutoring system where the AI doesn't follow rigid rules — it builds a probabilistic model of the student's mind and selects teaching actions that minimize uncertainty about what they know.

The core engine is a **POMDP (Partially Observable Markov Decision Process)** solved via **active inference** (Free Energy Principle). The system never directly observes a student's knowledge state — it infers it from behavior, then picks the next problem to simultaneously reduce its own uncertainty and maximize learning outcomes.

---

## How It Works

Every time a student answers a problem, the system observes three things:

| Observable | Values |
|---|---|
| Correctness | correct / incorrect |
| Response time | fast (<5s) / normal (5–15s) / slow (>15s) |
| Explanation quality | low / medium / high |

From these observations, it maintains a **belief distribution** over 5 hidden cognitive dimensions per skill:

| Dimension | Bins | What it captures |
|---|---|---|
| **Retrievability (RS)** | 5 | Current recall probability (0–1) |
| **Stability (SS)** | 4 | How deeply encoded the memory is (in days) |
| **Schema** | 3 | Knowledge organization: none / partial / full |
| **Working Memory Load** | 3 | Cognitive demand: low / moderate / high |
| **Affect** | 3 | Emotional state: frustrated / engaged / bored |

Joint state space per skill: **5 × 4 × 3 × 3 × 3 = 540 states**, tracked as a factorized belief (mean-field approximation).

### Action Selection via Expected Free Energy

The agent selects the next pedagogical action by minimizing **Expected Free Energy (EFE)**:

```
G(action) = epistemic_value + pragmatic_value
```

- **Epistemic value** — how much the action will reduce uncertainty about the student's state (drives diagnostic probing when the agent doesn't know enough)
- **Pragmatic value** — how well the action aligns with preferred outcomes (correct answers, good explanations, engaged student)

This naturally produces adaptive behavior: when uncertain → diagnose; when confident → teach optimally.

### Available Teaching Actions

| Action | When triggered |
|---|---|
| `reteach` | RS very low — retrieval will fail |
| `desirable_difficulty` | RS low — struggle zone for max encoding gain |
| `practice` | RS moderate — productive practice |
| `worked_example` | Schema undeveloped — full scaffolding |
| `faded_example` | Schema partial — gradually remove hints |
| `space_and_test` | High stability — spaced retrieval |
| `interleave` | Low discriminability — student confusing similar skills |
| `diagnostic_probe` | High uncertainty — need more information |

---

## Architecture

```
Frontend (React/TypeScript)
        ↓ HTTP (FastAPI)
    server.py
        ↓
  ┌─────────────────────────────────────┐
  │         active_inference/           │
  │  pomdp.py — ActiveInferenceAgent    │
  │    ├── A matrices (observation)     │
  │    ├── B matrices (transition)      │
  │    ├── C vectors (preferences)      │
  │    └── pymdp Agent (EFE solver)     │
  └─────────────────────────────────────┘
        ↓
  ┌─────────────────────────────────────┐
  │           inference/                │
  │  State estimators per dimension:    │
  │  retrievability, stability,         │
  │  schema, wm_load, affect,           │
  │  discriminability, confidence       │
  └─────────────────────────────────────┘
        ↓
  ┌─────────────────────────────────────┐
  │            domain/                  │
  │  Problem generation + taxonomy      │
  │  14 algebra skill types             │
  │  Knowledge space (540 problems)     │
  └─────────────────────────────────────┘
```

---

## Project Structure

```
adaptive-learning/
├── active_inference/        # POMDP core
│   ├── pomdp.py             # ActiveInferenceAgent (A/B/C/D matrices + pymdp)
│   ├── state_space.py       # 5-dim state space definition and discretization
│   └── transition_model.py  # How actions change cognitive states
│
├── inference/               # Per-dimension state estimators
│   ├── affect_state.py      # Frustration / engagement / boredom
│   ├── confidence.py        # Self-reported confidence tracking
│   ├── discriminability.py  # Confusion between similar skills
│   ├── memory_state.py      # Retrievability + stability (spaced repetition)
│   ├── schema_state.py      # Knowledge organization level
│   └── wm_load.py           # Working memory demand
│
├── domain/                  # Algebra problem generation
│   ├── taxonomy.py          # 11 skills, confusable pairs, difficulty tiers
│   ├── generate.py          # Problem generator
│   ├── render.py            # Problem rendering (LaTeX / plain text)
│   ├── knowledge_space.py   # Problem bank structure
│   └── cognitive_features.py # Feature extraction from student responses
│
├── simulation/              # Monte Carlo simulation of learner types
│   ├── learner_types.py     # Struggling / average / advanced learner models
│   ├── simulated_agent.py   # Simulated student responses
│   └── monte_carlo.py       # Batch simulation + analysis
│
├── frontend/                # React UI
│   └── src/
│       ├── App.tsx           # Session state machine (login→problem→confidence→feedback)
│       ├── components/
│       │   ├── ProblemCard.tsx      # Problem display + answer input
│       │   ├── ConfidencePrompt.tsx # Self-report confidence (1–5)
│       │   ├── FeedbackView.tsx     # Correctness + explanation
│       │   └── SessionDashboard.tsx # Live belief state visualization
│       └── api.ts            # Backend API client
│
├── server.py                # FastAPI server
├── meta_function.py         # Rule-based fallback (used if pymdp unavailable)
└── data/                    # Generated knowledge space JSON
```

---

## Running Locally

### Backend

```bash
# Install dependencies
pip install -r requirements.txt

# Start the API server
python server.py
# → http://localhost:8000
```

Requirements include: `fastapi`, `uvicorn`, `pymdp`, `jax`, `numpy`, `pydantic`

### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start dev server
npm run dev
# → http://localhost:5173
```

Make sure the backend is running first — the frontend connects to `localhost:8000` by default.

---

## Theoretical Background

This system applies the **Free Energy Principle** (Karl Friston) to education:

- The agent maintains beliefs about a student's latent cognitive state
- It selects actions (problem types) that minimize expected surprise
- Epistemic actions reduce uncertainty; pragmatic actions pursue learning goals
- The tension between these drives naturally adaptive, personalized teaching

The 5-dimensional state model is grounded in cognitive science:
- **Retrievability + Stability** → spaced repetition theory (Ebbinghaus / Wozniak)
- **Schema** → cognitive load theory (Sweller)
- **Working Memory** → Baddeley's model
- **Affect** → self-determination theory (Ryan & Deci)

---

## Tech Stack

| Layer | Technology |
|---|---|
| AI engine | [pymdp](https://github.com/infer-actively/pymdp) + JAX |
| Backend | FastAPI + Python |
| Frontend | React 19 + TypeScript + Vite |
| UI | Tailwind CSS + shadcn/ui + Framer Motion |
| Deployment | Vercel |
