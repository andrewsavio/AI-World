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
| **4** | Patterns | Rings · Checkerboard | Pass a real-data subject |
| **5** | Mastery | Flower · Spiral | — |
| **6** | Invention | *(none specified)* | Create a puzzle **or** discover a new sense |
| **7** | Open research | No fixed subjects | No gates |

### Promotion rules

A student advances when **both** conditions are met:
1. It has **passed every subject** of its current level (an exam on unseen examples, at or above the pass mark).
2. It has met the **extra gate** for that level (if any).

Puzzles are created at Level 6 and later: a student that reaches Level 6 starts making puzzles for the others to study (a world first is awarded to the first student to master each one). A new *sense* is discovered when a stuck student invents an input feature such as x², radius or angle and publishes it to the Library.

---

## Design References

| Decision | Reference |
|---|---|
| Easy-before-hard ordering | Bengio et al., *Curriculum Learning* (ICML 2009) |
| Subject ordering within levels | TensorFlow Playground dataset order (Smilkov et al., Google Brain) |
| Level names (Foundations → Mastery → Invention → Research) | Bloom's Revised Taxonomy of Educational Objectives |
| Education stage of each level (primary → doctoral) | UNESCO ISCED level descriptors |
| Invention and Open Research as terminal levels | Analogous to postgraduate research — no prescribed content |

---

## Real-Data Subjects (Level 4 gate)

Students must pass at least one subject that uses real-world data before leaving Level 4.
Available real-data subjects:

Students create these subjects themselves while exploring in the Internet Lab:

| Subject | Data source | Classification task |
|---|---|---|
| Earthquake depth | USGS Earthquake Hazards API (last 30 days) | Deeper than 30 km vs shallower, from longitude and latitude |
| Rain at a real place | Wikipedia (finds a real place) + Open-Meteo | Raining vs dry, from two weather measurements the student chooses |

The pass mark for each is set just below what a simple nearest-neighbours method scores on the same data, because real data is messy.

---

## What a student may choose

Inside its level a student picks its own next subject (curious students sometimes peek one level up), decides when it feels ready for an exam, and can give up on a subject that is too hard and come back later. The Syllabus view lists the levels, every subject, and which students are where.

---

## References

- Bengio, Y. et al. (2009). Curriculum Learning. *ICML*.
- TensorFlow Playground — https://playground.tensorflow.org
- Bloom, B. S. et al. (1956). *Taxonomy of Educational Objectives*. David McKay Company.
- UNESCO ISCED 2011 — https://uis.unesco.org/sites/default/files/documents/isced-2011-en.pdf
