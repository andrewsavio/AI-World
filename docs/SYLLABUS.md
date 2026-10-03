# 📜 AI World — Curriculum Design & Syllabus

This document explains the design, ordering and promotion logic of the seven-level curriculum used in AI World.

---

## Why a Levelled Curriculum?

Neural networks don't learn well when problems are too hard too soon.
**Curriculum learning** — presenting examples from easy to hard — is a well-studied technique that improves convergence speed and final accuracy (Bengio et al., 2009 — *Curriculum Learning*, ICML).

The levels below apply that principle to the students' lived experience: a student literally cannot move to the next level until it has proved it can handle the current one.

---

## The Seven Levels

| Level | Name | Subjects | Extra gate | Notes to leave |
|---|---|---|---|---|
| **1** | Foundations | Left vs Right · Top vs Bottom · Diagonal | — | 2 |
| **2** | Shapes | Stripe · Circle · Diamond | — | 4, 📘 Learner |
| **3** | Curves & Logic | Ellipse · XOR · Waves | — | 8, 📘 Learner |
| **4** | Patterns | Rings · Checkerboard | Pass a real-data subject | 12, 🎓 Expert |
| **5** | Mastery | Flower · Spiral | — | 16, 🎓 Expert |
| **6** | Invention | *(none specified)* | Create a puzzle **or** invent a sense | 20, 🏆 Master |
| **7** | Open research | No fixed subjects | No gates | — |

### Promotion rules

A student advances when **all three** conditions are met:
1. It has **passed every subject** of its current level (exam accuracy ≥ threshold).
2. It has met the **extra gate** for that level (if any).
3. Its notebook contains at **least N notes** (table column above).

---

## Design References

| Decision | Reference |
|---|---|
| Easy-before-hard ordering | Bengio et al., *Curriculum Learning* (ICML 2009) |
| Subject ordering within levels | TensorFlow Playground dataset order (Smilkov et al., Google Brain) |
| Level names (Foundations → Mastery → Invention → Research) | Bloom's Revised Taxonomy of Educational Objectives |
| Note counts as a proxy for knowledge depth | UNESCO ISCED level descriptors |
| Invention and Open Research as terminal levels | Analogous to postgraduate research — no prescribed content |

---

## Real-Data Subjects (Level 4 gate)

Students must pass at least one subject that uses real-world data before leaving Level 4.
Available real-data subjects:

| Subject | Data source | Classification task |
|---|---|---|
| Earthquake magnitude | USGS Earthquake Hazards API | Magnitude ≥ 5.0 vs < 5.0 |
| Temperature anomaly | Open-Meteo Historical API | Above-average vs below-average day |

---

## Open Curriculum

Beyond the fixed syllabus, every level has an **open curriculum** track.
When a student chooses to browse or research freely, it writes notes to its own notebook and the topic is recorded as an open-curriculum entry.
The open curriculum is displayed in the Syllabus view alongside the fixed one.

---

## References

- Bengio, Y. et al. (2009). Curriculum Learning. *ICML*.
- TensorFlow Playground — https://playground.tensorflow.org
- Bloom, B. S. et al. (1956). *Taxonomy of Educational Objectives*. David McKay Company.
- UNESCO ISCED 2011 — https://uis.unesco.org/sites/default/files/documents/isced-2011-en.pdf
