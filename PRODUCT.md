# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Three audiences use the same interface, all confirmed:

- **The creator (Andrew Savio)** runs the world every day, usually in the background, checks on the students, reads the dashboard, founds buildings, and decides when there is something worth showing.
- **Recruiters and engineers** arrive from LinkedIn posts, videos or the GitHub repository, often for a few minutes, to judge the creator's machine-learning and engineering skills.
- **Curious learners** (students, newcomers to AI) explore it to see how neural networks and reinforcement learning actually work.

## Product Purpose

AI World is a 3D planet where 10 tiny neural networks live as students: they choose what to do, study subjects, sit exams on unseen data, get promoted through a seven-level syllabus that ends in quantum computing, run their own experiments, and teach each other.

Success: **someone who uses it understands how AI learns.** Backpropagation, rewards and reinforcement learning, curricula, generalisation to unseen data and the differences between learners are visible and explainable, not hidden behind a demo.

## Positioning

The students genuinely learn: every decision and lesson comes from small PyTorch networks whose weights change while you watch (a skill brain trained by backpropagation, a decision brain trained by REINFORCE). There is no language model role-playing the agents, unlike LLM-driven agent worlds. Everything is inspectable: every connection, every exam, every reward, and a dashboard that explains why one student is ahead of another.

## Operating Context

- Runs locally: `start-world-background.bat` (or `python server.py`) serves the page at `http://localhost:8000`; a tray launcher can start it at Windows login with the window minimized so the students keep living for days.
- The world is saved in `saves/` (students, subjects, news, brain weights) every 2 minutes, when the tab closes, and on demand; reopening continues the same world.
- Time follows the user's real local clock: a world day is an Earth day, and students sleep at night.
- Progress is followed through the 📊 Dashboard (including "Who's ahead, and why"), the 📜 Syllabus, the news and thoughts feeds, the brain panels and `saves/history.csv` (one row per day).
- The creator films short videos of the world (Hyperframes) and posts episodes on LinkedIn.

## Capabilities and Constraints

- **Learning:** skill brain (MLP, mini-batch SGD with backprop, grows neurons, gains input "senses"); decision brain (policy network, REINFORCE, residual on top of needs and personality); curriculum Levels 1–4 classic classification maps, Levels 5–7 quantum computing learned from an exact two-qubit simulator; a Quantum Lab where students design their own experiments; teaching credit when another student learns from an invention.
- **Economy:** students earn coins; the class founds buildings (Data Centre for compute, Data Lab, Lecture Hall, Hall of Discoveries). Individual wallets and a store are proposed, not built.
- **Graphics:** Intel integrated graphics is the baseline target. Richer, more realistic visuals are allowed as long as the Low quality setting stays smooth on weak laptops.
- **Internet:** the students may use the internet for some things (read-only public sources such as Wikipedia). Today they use none: the Quantum Lab replaced web search. Internet use never posts, signs up or sends user data.
- **Hosting:** a hosted version that really runs, for recruiters to try, is a goal. Today the hosted copy (https://tensorworld.netlify.app/) only explains itself unless pointed at a Python server with `?api=`.
- **No language model** anywhere in the students or the world.
- **Honest claims only:** students cannot invent anything new to humanity; discoveries are labelled "new to the AI World". Correlations are not presented as causes.
- **No AI-assistant or tool-vendor mentions** in the project's files, README or commits.
- Stack: single `index.html` (three.js from a CDN, no build step), `server.py` (Python standard library), `brain.py` (PyTorch, CPU), `launcher.py` (tray); pytest suite and CI on GitHub Actions.

## Brand Commitments

- Name: **AI World** (the page title reads "AI World Tycoon"). Repository: https://github.com/andrewsavio/AI-World (MIT licence).
- The 10 students and their names: Ada, Alan, Grace, Hedy, Marie, Nikola, Rosalind, Kurt, Emmy, Srini.
- Voice: plain, honest and explanatory, written for people new to AI; jargon is explained when it is used.

## Evidence on Hand

- Screenshots in `docs/screenshots/` (some outdated after the graphics changes).
- Episode 1 launch videos in `brag-output/` and `brag-output-2026-10-06-213055/` (narrated, with captions and LinkedIn post text); not committed to git.
- Real run data: `saves/world.json`, `saves/brains.pt`, `saves/history.csv`, and the dashboard's before/after numbers (for example average liking for studying moved from −0.22 to +0.39 after the reward fix).
- 33 automated tests (`tests/test_brain.py`) and the `python brain.py` self-test.
- Absent, never to be fabricated: testimonials, user counts, press coverage, benchmarks against other systems, and any real-world invention by the students.

## Product Principles

1. **Learning must be visible.** Every feature should make how the students learn easier to see or explain; spectacle that hides the mechanism works against the product.
2. **Real, not staged.** Show genuine learning and real data; never fake progress, discoveries or numbers.
3. **Explain for newcomers, reward experts.** A first-timer can follow the story, and an ML engineer can verify the details underneath.
4. **Runs on an ordinary laptop.** Quality scales up, but a weak machine must still run the world smoothly.
5. **The world keeps living.** Saving, background running and real time make it a world that persists, not a demo that resets.
