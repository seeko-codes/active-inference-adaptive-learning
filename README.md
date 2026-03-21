# Synthetic Students: Normative Education via Active Inference

> **Meta question:** Can we normatively determine optimal curricula by simulating synthetic learners — without collecting real-world student data?

Traditional learning design relies on trial and error across real students. This project takes the opposite approach: model the cognitive processes mathematically, simulate a wide range of learner types, and let the optimal curriculum emerge from the model.

We run Monte Carlo simulations of **active inference agents** — synthetic students whose hidden knowledge states evolve as they solve algebra problems. By mapping the full space of possible learning paths across simulated learner profiles, we identify which sequences work best for each cognitive type. The result is a proof-of-concept for **normative education**: curricula that are mathematically derived, not empirically guessed.

---

## The Two Layers

### Layer 1 — Simulation (the meta question)
Synthetic students are modeled as POMDPs. Each agent's knowledge state Θ is hidden and inferred from observable responses:

| State Dimension | What it tracks |
|---|---|
| **Bc** — Conceptual knowledge | Understanding of algebraic structure |
| **Bp** — Procedural fluency | Execution accuracy |
| **Korg** — Organization | How knowledge is structured and connected |
| **φ (RS/SS)** — Memory dynamics | Retrievability + stability (spaced repetition) |
| **WM** — Working memory load | Cognitive demand in the moment |
| **Affect** | Frustrated / engaged / bored |

Monte Carlo simulation runs n agents through the 2,519-problem algebraic knowledge space and maps which learning paths produce the best belief state evolution for each learner type.

**Results so far:**
- *Student A (Weak Procedural)* — system detects Bc/Bp dissociation, shifts to procedural drilling
- *Student B (Rote Learner)* — system forces conceptual repair through property identification tasks
- *Student C (Memory Decay)* — system auto-implements spacing via detected φ decline
- Posterior beliefs narrow to precise state estimates after 15–20 problems

### Layer 2 — Live System (the normal question)
The same active inference agent that drives simulation runs in real time as a tutor. Every student response updates the belief state; the agent then selects the next teaching action by minimizing **Expected Free Energy**:

```
G(action) = epistemic value + pragmatic value
```

- High uncertainty → epistemic dominates → diagnostic probing
- Low uncertainty → pragmatic dominates → deploy optimal technique

**Teaching actions drawn from learning science:**

| Action | Grounded in |
|---|---|
| `worked_example` → `faded_example` → `full_problem` | Cognitive Load Theory (Sweller, 1988) |
| `space_and_test` | Spacing + desirable difficulties (Bjork, 1994) |
| `interleave` | Discrimination learning, transfer theory |
| `desirable_difficulty` | Bjork & Bjork (1992) new theory of disuse |
| `diagnostic_probe` | Knowledge Tracing (Corbett & Anderson, 1995) |

---

## Architecture

```
Monte Carlo Simulation                   Live Tutor
─────────────────────                    ──────────
n synthetic agents                       Real student
       ↓                                      ↓
  POMDP belief update  ←── same engine ───→  POMDP belief update
       ↓                                      ↓
  Policy via EFE                         Policy via EFE
       ↓                                      ↓
  Learning path logged                   Next problem served
       ↓
  Normative landscape map
```

```
active_inference/
  pomdp.py            # A/B/C/D matrices + pymdp EFE solver
  state_space.py      # State dimensions and discretization
  transition_model.py # How actions change cognitive states

inference/            # Per-dimension state estimators
  memory_state.py     # φ: retrievability + stability
  schema_state.py     # Korg: knowledge organization
  wm_load.py          # Working memory demand
  affect_state.py     # Frustration / engagement / boredom
  discriminability.py # Confusion between similar skills

simulation/           # Monte Carlo normative mapping
  learner_types.py    # Struggling / average / advanced profiles
  simulated_agent.py  # Synthetic student response model
  monte_carlo.py      # Batch simulation + landscape analysis

domain/               # Algebraic knowledge space
  taxonomy.py         # 11 skills, 10 question types, confusable pairs
  knowledge_space.py  # 2,519 unique problem forms
  generate.py / render.py

frontend/             # React/TypeScript tutor UI
server.py             # FastAPI — bridges inference engine to UI
```

---

## Running

**Backend**
```bash
pip install -r requirements.txt
python server.py          # → localhost:8000
```

**Frontend**
```bash
cd frontend && npm install && npm run dev   # → localhost:5173
```

**Simulation**
```bash
python -m simulation.monte_carlo           # runs normative mapping
```

---

## References

- Parr, Pezzulo & Friston (2022). *Active Inference: The Free Energy Principle in Mind, Brain, and Behavior.* MIT Press.
- Corbett & Anderson (1995). Knowledge tracing: Modeling procedural knowledge acquisition.
- Bjork (1994). Institutional impediments to self-directed learning.
- Bjork & Bjork (1992). A new theory of disuse and stimulus fluctuation.
- Sweller (1988). Cognitive load during problem solving.
- Piech et al. (2015). Deep knowledge tracing.
- Singley & Anderson (1989). *The Transfer of Cognitive Skill.*
