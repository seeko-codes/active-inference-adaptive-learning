"""
Simulation layer for normative education validation.

Answers the meta-question: "How do we test that the system works
without needing thousands of students?"

Modules:
- learner_types: Archetypal and randomly sampled learner definitions
- simulated_agent: Ground-truth cognitive model (the generative process)
- monte_carlo: N agents x M types x K policies simulation engine
- analysis: Parameter recovery, policy comparison, state ablation
"""
