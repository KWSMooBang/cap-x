"""Agent as BG Generator (Belief Graph Ver3): benchmark-independent Belief Graph generation from a
demonstration's video + robot state + instruction. See ``capx/bggen/README.md``.

Generator modules (records, events, perception, tracks, geometry, graph, verifier, agent, prompts, models,
generator, backward, serialize, pipeline) never read ground truth; ``oracle`` and ``evaluate`` are the
evaluation side; ``adapters`` turn benchmark data into both.
"""
