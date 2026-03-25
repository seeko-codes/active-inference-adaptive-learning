# Poster Text — Synthetic Students: A New Approach for Educational Design

## TITLE

**Synthetic Students: A Normative Approach to Adaptive Educational Design**

*Author Name* | Department of Mathematics, University of Central Florida
Faculty Mentor: *[Name]*, *[Department]*

---

## 1. INTRODUCTION

Traditional methods of learning design rely on trial and error with real students — demanding large amounts of data, raising ethical questions, and producing heuristic curricula that are not individually optimized.

We propose a **normative method**: model the cognitive processes of the learner from first principles, simulate thousands of synthetic students, and **deductively determine the most effective learning path for each learner type** — without conducting a single real-world experiment.

By running Monte Carlo simulations across diverse learner populations, we normatively map the landscape of possible learning paths to determine which instructional strategies are most effective for each type of student.

---

## 2. METHODS

Our method models student knowledge as a hidden variable in a **Partially Observable Markov Decision Process (POMDP)** with five cognitive dimensions:

- **Retrieval Strength (RS)** — can the student access what they learned?
- **Storage Strength (SS)** — how durable is the memory trace?
- **Schema Level** — how well has the student organized the material?
- **Working Memory Load** — is the student cognitively overloaded?
- **Affect** — is the student frustrated, engaged, or bored?

The system selects from 8 teaching actions (test, reteach, worked example, interleave, etc.) by minimizing **expected free energy** — an objective that naturally balances *exploring* what the student knows with *teaching* what they need.

We compare 5 instructional policies across **[N_STUDENTS] synthetic students** sampled from empirically grounded cognitive distributions (Cowan 2001, Ashcraft 2002, FSRS):

| Policy | Strategy |
|--------|----------|
| **Active Inference** | Full Bayesian model, EFE minimization |
| **Meta-Function** | Hand-crafted rules from learning science |
| **FSRS-Only** | Adapts on memory state only |
| **Fixed Curriculum** | Textbook order, no adaptation |
| **Random** | Control baseline |

*[SYSTEM DIAGRAM — keep existing but smaller]*

---

## 3. RESULTS

**[PLACEHOLDER — fill after precompute]**

**Policy Comparison** ([N_STUDENTS] students, [N_SESSIONS] sessions, [N_PROBLEMS] problems each)

*[BAR CHART: Mean final mastery by policy with error bars]*

- Active inference achieves **[X.XXX]** mean mastery vs. meta-function **[X.XXX]** (+[X.X]%)
- Both adaptive policies significantly outperform fixed curriculum (**[X.XXX]**) and random (**[X.XXX]**)

**Learning Trajectories**

*[LINE CHART: Mastery over sessions for each policy, one representative learner type]*

- Active inference shows steeper early learning and better retention after forgetting intervals
- Fixed curriculum plateaus after session [N] — no adaptation to student state

**Learning Landscape**

*[HEATMAP: Policy × Learner Type → mastery]*

- Largest advantage for [LEARNER_TYPE] learners: +[X.X]% over next best policy
- [LEARNER_TYPE] students show greatest sensitivity to policy choice (vulnerability = [X.XXX])

**Parameter Recovery**

- The system correctly identifies student schema level from observations with **[XX.X]%** accuracy
- RS inference error: **[X.XXX]** mean absolute error

---

## 4. CONCLUSION

This work demonstrates a proof of concept for **normative education** — learning curricula mathematically optimized for each learner type. By modeling mastery as a multidimensional hidden state, curriculum selection becomes a consequence of the model rather than a design choice. This architecture is domain-general: applicable to any structured domain where mastery can be decomposed into cognitive dimensions.

---

## 5. FUTURE WORK

- **Pilot study:** Calibrate the model against real student data in an introductory algebra course
- **Parameter recovery:** Validate inference accuracy from synthetic observation data
- **Identifiability frontier:** Determine how additional observation types (gaze tracking, confidence reports) improve state estimation

---

## 6. REFERENCES

1. Parr, T., Pezzulo, G., & Friston, K. J. (2022). *Active Inference: The Free Energy Principle in Mind, Brain, and Behavior.* MIT Press.
2. Bjork, R. A. (1994). Institutional impediments to self-directed learning. *From Learning Processes to Cognitive Processes.*
3. Sweller, J. (1988). Cognitive load during problem solving. *Cognitive Science, 12*(2), 257-285.
4. Piech, C., et al. (2015). Deep knowledge tracing. *Advances in Neural Information Processing Systems, 28.*
5. Corbett, A. T., & Anderson, J. R. (1995). Knowledge tracing. *User Modeling and User-Adapted Interaction, 4*(4), 253-278.
6. Kornell, N., & Bjork, R. A. (2008). Learning concepts and categories. *Psychological Science, 19*(6), 585-592.

---

## 7. ACKNOWLEDGEMENTS

This research was supported by the University of Central Florida Office of Undergraduate Research. The authors thank the researchers whose foundational work made this project possible, particularly in active inference, learning science, and knowledge tracing.

---

## PLACEHOLDER KEY

After running the precompute, search-and-replace these:

- `[N_STUDENTS]` — total students simulated (e.g., 10,000)
- `[N_SESSIONS]` — sessions per trajectory (5)
- `[N_PROBLEMS]` — problems per session (20)
- `[X.XXX]` — specific mastery/accuracy values from results
- `[X.X]%` — percentage improvements
- `[LEARNER_TYPE]` — specific archetype names from landscape analysis
- `[BAR CHART]`, `[LINE CHART]`, `[HEATMAP]` — generated figures
