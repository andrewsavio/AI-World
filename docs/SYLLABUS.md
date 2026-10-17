# 📜 AI World — Curriculum Design & Syllabus

This document explains the design, ordering and promotion logic of the seven-level curriculum used in AI World.

---

## Why a Levelled Curriculum?

Neural networks don't learn well when problems are too hard too soon.
**Curriculum learning** — presenting examples from easy to hard — is a well-studied technique that improves convergence speed and final accuracy (Bengio et al., 2009 — *Curriculum Learning*, ICML).

The levels below apply that principle to the students' lived experience: a student literally cannot move to the next level until it has proved it can handle the current one.

---

## The Seven Levels

| Level | Name | Subjects | Extra gate |
|---|---|---|---|
| **1** | Foundations | Left vs Right · Top vs Bottom · Diagonal | — |
| **2** | Shapes | Stripe · Circle · Diamond | — |
| **3** | Curves & Logic | Ellipse · XOR · Waves | — |
| **4** | Patterns | Rings · Checkerboard (Flower · Spiral optional) | Pass an experiment of its own from the Quantum Lab |
| **5** | Qubits | Superposition · Interference · Measurement basis | — |
| **6** | Entanglement & algorithms | Entanglement · Bell correlations · Grover search · Decoherence | Discover a circuit **another student learns from** |
| **7** | Quantum research | No fixed subjects | No gates; the goal is to teach others 5 times |

### Promotion rules

A student advances when **both** conditions are met:
1. It has **passed every subject** of its current level (an exam on unseen examples, at or above the pass mark).
2. It has met the **extra gate** for that level (if any).

Circuits are discovered at Level 6 and later: a student searches for a two-qubit circuit whose outcome forms a clear, learnable pattern and shares it for the others to study (a world first is awarded to the first student to master each one). A new *sense* is discovered when a stuck student invents an input feature such as x², radius or angle and publishes it to the Library.

---

## Design References

| Decision | Reference |
|---|---|
| Easy-before-hard ordering | Bengio et al., *Curriculum Learning* (ICML 2009) |
| Subject ordering within levels | TensorFlow Playground dataset order (Smilkov et al., Google Brain) |
| Level names (Foundations → Mastery → Invention → Research) | Bloom's Revised Taxonomy of Educational Objectives |
| Education stage of each level (primary → doctoral) | UNESCO ISCED level descriptors |
| Quantum track for Levels 5–7 | Standard first topics of quantum computing courses: single qubits, measurement, entanglement, Bell tests, Grover's algorithm, decoherence |

---

## The quantum track (Levels 5–7)

All quantum data comes from a small exact simulator of two qubits (4 complex amplitudes; gates H, X, RX, RY, RZ and CNOT). Each subject is a map: two settings in, one yes/no outcome out. The students are never shown the formulas.

| Subject | Level | The experiment | What must be predicted |
|---|---|---|---|
| Superposition | 5 | RY(a) then RZ(b) on one qubit | Does it read 1 more often than 0? (the phase b is invisible to this measurement) |
| Interference | 5 | RY(a), H, RZ(b), H | Does it read 1? (an interferometer) |
| Measurement basis | 5 | RY(a), RZ(b), then measured in the X basis | Does it read 1 ("−")? |
| Entanglement | 6 | RY(a) and RY(b) on two qubits, then CNOT | Is the concurrence above 0.5? |
| Bell correlations | 6 | A Bell pair read at detector angles a and b | Do the two results agree more often than not? |
| Grover search | 6 | k iterations over N items | Does it find the marked item more than half the time? |
| Decoherence | 6 | A precessing qubit losing its phase to noise | Is it still mostly in "+"? |

Each pass mark is set just below what a simple nearest-neighbours method scores on the same map, at most 92%.

## The Quantum Lab (Level 4 gate)

Students must pass an experiment of their own before leaving Level 4. A student builds a random circuit of 3–5 gates on two qubits with two settings, chooses what to measure (a qubit reading 1, the qubits agreeing, entanglement) and runs it 1,500 times. If the settings decide the outcome in a learnable way, the experiment becomes its own subject; otherwise it is rejected honestly. Students learn which measurements tend to work (an upper-confidence-bound rule over their own record). Experiments are never shared; circuits discovered at Level 6+ are.

---

## What a student may choose

Inside its level a student picks its own next subject (curious students sometimes peek one level up), decides when it feels ready for an exam, and can give up on a subject that is too hard and come back later. The Syllabus view lists the levels, every subject, and which students are where.

---

## References

- Bengio, Y. et al. (2009). Curriculum Learning. *ICML*.
- TensorFlow Playground — https://playground.tensorflow.org
- Bloom, B. S. et al. (1956). *Taxonomy of Educational Objectives*. David McKay Company.
- UNESCO ISCED 2011 — https://uis.unesco.org/sites/default/files/documents/isced-2011-en.pdf
