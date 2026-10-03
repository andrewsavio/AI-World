# The AI World syllabus

This document explains *why* the students learn things in the order they do, and how promotion works. The numbers live in the `SYLLABUS` and `SKILLS` constants in `index.html`, so a contribution can change the syllabus by editing data, not logic.

## Two syllabi side by side

1. **The levels (fixed).** Seven levels of subjects, with promotion between them. This is what this page describes.
2. **The open curriculum (free).** Each student's language mind chooses what to read on Wikipedia and writes notes in its own notebook. Nothing restricts the topics. The dashboard records what they actually chose and compares it with a menu of 20 suggested areas of knowledge. This is the "knowledge track" that the levels also ask for.

## The levels

| Level | Name | Subjects | Also needed | Knowledge to leave | Stage (ISCED) | Thinking (Bloom) |
|---|---|---|---|---|---|---|
| 1 | Foundations | Left vs Right, Top vs Bottom, Diagonal | | 2 notes | Primary (1) | Remember |
| 2 | Shapes | Stripe, Circle, Diamond | | 4 notes, Learner | Lower secondary (2) | Understand |
| 3 | Curves & Logic | Ellipse, XOR, Waves | | 8 notes, Learner | Upper secondary (3) | Apply |
| 4 | Patterns | Rings, Checkerboard | pass a real-data subject | 12 notes, Expert | Bachelor (6) | Analyse |
| 5 | Mastery | Flower, Spiral | | 16 notes, Expert | Master (7) | Evaluate |
| 6 | Invention | | create a puzzle or invent a sense | 20 notes, Master of a field | Doctoral (8) | Create |
| 7 | Open research | no gates | | | beyond the degree | Create |

Subjects are classification tasks: the student sees a point on a map and must say whether it is blue or red. "Real-data subjects" are created by the students themselves from live data (USGS earthquakes, Open-Meteo weather) and are available from Level 4. Puzzles invented by students are available from Level 6.

## Promotion rules

A student is promoted when `gate(a).ready` is true:

- every subject listed for its level has been **passed** (an exam on unseen examples at or above the pass mark),
- the **extra requirement** is met (a real-data subject at Level 4; a created puzzle or invented sense at Level 6),
- the **knowledge requirement** is met: enough notes in its notebook and a high enough rank in one field (Curious, Learner, Expert, Master). Ranks come from quiz exams answered using only the student's own notes. The knowledge requirement is waived when the language mind (Ollama) or the internet is switched off, so the maths syllabus can still be completed on its own.

On promotion the student earns coins, and the first student to reach each level earns a bonus and a world-first headline. When a student has finished its exams but something else blocks promotion (for example it still needs notes), its decisions are nudged towards the missing thing.

## Why this order?

- **Easy before hard (curriculum learning).** Bengio et al. showed that presenting examples in a meaningful order, from easier to harder, can improve how neural networks learn ("Curriculum Learning", ICML 2009). Each level here is a bin of similar difficulty, and a student only moves on once it has mastered the current one. We measured that the harder subjects really need a bigger brain (a network with one small hidden layer could not learn Circle, Diamond, Ellipse or XOR in 30,000 lessons, but a network with two hidden layers could), so students have to grow neurons and invent senses as they climb, which is the point of the game.
- **The classic teaching order for small networks.** TensorFlow Playground, a widely used teaching tool, orders its datasets from linearly separable blobs to circle, XOR and finally spiral, and lets learners add input features such as x², y² and x·y. Our levels follow the same progression, and students gain the same kind of "senses".
- **Education stages (UNESCO ISCED 2011).** ISCED defines levels from primary (1), lower secondary (2) and upper secondary (3) through tertiary stages up to doctoral (8). The knowledge requirements grow in the same way: more notes and a deeper rank in a field at each stage.
- **Kinds of thinking (revised Bloom's taxonomy).** Remember, Understand, Apply, Analyse, Evaluate, Create. Early levels ask the student to reproduce a pattern, middle levels to use new senses and test them on real data, and the final levels to create something new.

## Changing the syllabus

- Add a subject: add an entry to `SKILLS` (name, `level`, `tier`, and the rule `f(x, y)`), then list its name in the right `SYLLABUS` level. Add a one-line description to `CORE_ABOUT`.
- Check it is learnable: train a small network on it with `brain.py` and see how many lessons it needs (the self-test at the bottom of `brain.py` shows how).
- Change a gate: edit the level's `skills`, `notes`, `rank`, `real` or `create` fields.

## References

- Bengio, Louradour, Collobert, Weston. *Curriculum Learning*. ICML 2009. <https://ronan.collobert.com/pub/2009_curriculum_icml.pdf>
- TensorFlow Playground. <https://playground.tensorflow.org>
- UNESCO Institute for Statistics. *International Standard Classification of Education (ISCED) 2011*. Summary: <https://www.cedefop.europa.eu/en/tools/vet-glossary/glossary/internationale-standardklassifikation-im-bildungswesen-isced-2011>
- Anderson and Krathwohl. *A Taxonomy for Learning, Teaching, and Assessing* (the 2001 revision of Bloom's taxonomy). Overview: <https://en.wikipedia.org/wiki/Bloom%27s_taxonomy>
