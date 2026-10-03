<div align="center">

<pre>
 █████╗ ██╗    ██╗    ██╗    ██╗ ██████╗ ██████╗ ██╗     ██████╗
██╔══██╗██║    ██║    ██║    ██║██╔═══██╗██╔══██╗██║     ██╔══██╗
███████║██║    ██║    ██║ █╗ ██║██║   ██║██████╔╝██║     ██║  ██║
██╔══██║██║    ██║    ██║███╗██║██║   ██║██╔══██╗██║     ██║  ██║
██║  ██║██║    ██║    ╚███╔███╔╝╚██████╔╝██║  ██║███████╗██████╔╝
╚═╝  ╚═╝╚═╝    ╚═╝     ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═════╝
</pre>

### *A living, breathing 3D planet where tiny neural networks go to school, fall in love with curiosity, and learn to think — one synapse at a time.*

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-CPU-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![three.js](https://img.shields.io/badge/three.js-WebGL-000000?style=for-the-badge&logo=threedotjs&logoColor=white)](https://threejs.org)
[![Ollama](https://img.shields.io/badge/Ollama-Gemma%202-111111?style=for-the-badge)](https://ollama.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-f59e0b?style=for-the-badge)](CONTRIBUTING.md)
[![Netlify](https://img.shields.io/badge/Netlify-Ready-00C7B7?style=for-the-badge&logo=netlify&logoColor=white)](https://netlify.com)

<img src="docs/screenshots/world-overview.jpg" alt="The AI World — a low-poly planet seen from orbit" width="820">

> **You are not playing a game. You are watching real machine learning happen — live, in 3D, in your browser.**

</div>

---

## 🌍 The Story

Somewhere on a tiny low-poly planet, ten artificial minds just woke up.

They have **needs** — hunger, fun, friendships. They have **personalities** — some are bold risk-takers, some are cautious scholars. They have a **school**, a **university**, a **research institute**, a **library**, an **exam hall**, a **dormitory**, an **internet lab** and a **park**.

Nobody tells them what to do.

They choose their own path. They search the real internet. They run real experiments on real earthquake and weather data. They invent new ways to sense the world. And each one thinks with a real local LLM (Gemma 2 via Ollama): it picks what to learn next, writes notes in its own notebook, and is quizzed on them.

Click on any student. Watch its neurons fire. Watch its weights shift. Watch it *think*.

---

## ✨ What Makes This Different

| | Feature | Detail |
|---|---|---|
| 🧠 | **Real neural networks** | Not simulated — actual PyTorch MLP models trained with backpropagation right in your browser session |
| 🎲 | **True free will** | Students have needs, personalities and learned preferences that drive every decision |
| 🌐 | **Live internet access** | They search **Wikipedia**, test hypotheses on **USGS earthquake** and **Open-Meteo** weather data |
| 🎓 | **A real language mind** | A local **Gemma 2 2B** LLM lets each student choose topics, write notes and invent new input senses as math formulas |
| 📜 | **A seven-level syllabus** | Students are **promoted** level by level, from straight lines to inventing new puzzles, plus an open curriculum of whatever they freely choose to learn ([details](docs/SYLLABUS.md)) |
| 🌅 | **Gorgeous on integrated graphics** | Day/night cycle, atmosphere, real interior shadows, auto-resolution scaling for 60 fps on iGPUs |

<div align="center">
<img src="docs/screenshots/inside-library-sunset.jpg" alt="Inside the library at golden hour" width="49%">
<img src="docs/screenshots/syllabus.jpg" alt="The live curriculum board" width="49%">
</div>

---

## 🧬 The Science Inside

Each student carries **two brains** built from scratch in PyTorch:

```
┌─────────────────────────────────────────────────────────────┐
│  SKILL BRAIN  (a few dozen connections)                     │
│  Trained by:  mini-batch SGD + backpropagation              │
│  Task:        learn to separate blue dots from red dots      │
│               on 2-D maps of growing complexity              │
│  Superpower:  can grow new neurons · can invent new senses  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  DECISION BRAIN  (policy network)                           │
│  Trained by:  REINFORCE (policy-gradient RL)                │
│  Task:        learn which activity maximises long-term      │
│               wellbeing from world rewards                  │
└─────────────────────────────────────────────────────────────┘
```

**Gemma 2 2B** (via Ollama) is every student's shared mind: it reads Wikipedia, writes notes, sets quizzes, and invents new mathematical **input features** (validated by a tiny safe parser that never calls `eval`) that help students sense the world in new ways.

---

## 🏗️ Architecture

```mermaid
flowchart LR
    subgraph Browser["🖥️ Browser  ·  index.html + three.js"]
        W["3D World + Simulation<br/>needs · personality · activities"]
        UI["Dashboard · Syllabus · Brain Viewer"]
    end
    subgraph Server["🐍 server.py  (Python stdlib HTTP)"]
        B["brain.py  ·  PyTorch<br/>skill brain · decision brain"]
        N["Notebooks  ·  JSON on disk"]
    end
    O["🦙 Ollama · Gemma 2<br/>Shared student minds"]
    I["🌐 Wikipedia · USGS · Open-Meteo<br/>(read-only GET)"]

    W <-->|"train · exam · act · reward"| B
    W --> N
    W <-->|"prompts"| O
    W -->|"GET only"| I
    UI --- W
```

| Layer | Technology | What it does |
|---|---|---|
| **Skill brain** | PyTorch MLP · autograd · mini-batch SGD | Learns to separate blue / red dots; grows neurons; gains new senses |
| **Decision brain** | PyTorch MLP · REINFORCE | Learns corrections to built-in instincts from world rewards |
| **Student minds** | Gemma 2 2B via Ollama | Chooses topics · writes notes · sets exams · invents input formulas |
| **Memory** | JSON notebooks + keyword RAG | Per-student knowledge (60-note cap → natural forgetting) |
| **3D World** | three.js · instancing · merged meshes · canvas textures | Low-poly town · avatars · day/night · atmosphere |
| **Server** | Python `http.server` · keep-alive · Origin checks | Serves the page · hosts brains · saves notebooks |

> Everything runs **entirely on your machine**. No cloud. No API keys. No telemetry.

---

## 🚀 Quick Start

### Requirements

- **Python 3.10+**
- **[Ollama](https://ollama.com/download)** *(optional — world runs without it)*
- A modern browser (Chrome / Edge recommended)

### Windows (30 seconds)

```bash
git clone https://github.com/andrewsavio/AI-World.git
cd AI-World
pip install -r requirements.txt
start-world.bat
```

The first run downloads **Gemma 2 2B** (~1.6 GB) through Ollama, then opens **http://localhost:8000** automatically.

### macOS / Linux

```bash
git clone https://github.com/andrewsavio/AI-World.git
cd AI-World
pip install -r requirements.txt
ollama pull gemma2:2b && ollama cp gemma2:2b aiworld-gemma   # once
python server.py
```

> **PyTorch is required.** The page has no brain of its own: if the Python server isn't running it pauses, shows a banner saying what to start, and reconnects by itself (even if Python restarts mid-run).
>
> **Ollama is optional.** Without it the students have no language mind and no free-text learning, but the maths syllabus and the real-data experiments still work.

---

## 🎮 Controls

| Action | How |
|---|---|
| 🌍 Rotate / zoom | drag · scroll |
| 🏛️ Look inside a building | click it, or use the camera buttons under the view |
| 🧠 Read a student's mind | click the student avatar, or its row in the table |
| 📜 Syllabus / 📊 Dashboard | header buttons |
| 🎨 Quality | `Auto` (recommended) · `High` · `Low` — menu in the 3D view |
| ⏹️ Stop everything | **Stop world** button, or the tray icon in background mode |

---

## 🗂️ Repository Layout

```
AI-World/
├── index.html              ← Entire front end: 3D world, simulation, UI
│                             (single file, ES modules from CDN)
├── brain.py                ← PyTorch brains + safe formula parser
│                             run `python brain.py` for the self-test
├── server.py               ← HTTP server: static, /brain/*, /policy/*, notebooks
├── launcher.py             ← Windows tray-icon launcher
├── export_dataset.py       ← Export student notes → fine-tuning JSONL
├── requirements.txt        ← torch  (that's the only pip dependency)
├── models/
│   └── Modelfile           ← Ollama model definition for Gemma
├── start-world.bat         ← Windows one-click launcher
├── start-world-background.bat
└── docs/screenshots/       ← Images used in this README
```

---

## 🌐 Netlify / Static Hosting

The front end is a **single self-contained `index.html`** that loads three.js from a CDN, so any static host can serve it.

> **Note:** the students' brains run in Python (PyTorch), so a static host on its own **cannot run the world**: the page shows an *OFFLINE* banner until it can reach `server.py`. To host it online you need somewhere that runs Python (for example Render, Railway, a VPS or a Hugging Face Space). Pointing a Netlify-hosted front end at a remote Python backend is not supported yet, which makes it a good first issue.

**Deploy to Netlify in one click:**

[![Deploy to Netlify](https://www.netlify.com/img/deploy/button.svg)](https://app.netlify.com/start/deploy?repository=https://github.com/andrewsavio/AI-World)

Or drag-and-drop `index.html` into [app.netlify.com/drop](https://app.netlify.com/drop).

**AI World is local-first.** The real world runs on your computer (`start-world.bat`). A hosted copy is only a *representation*: it shows an explanation instead of running. To make a hosted page talk to a Python server you run yourself, open it with `?api=https://your-server` and start that server with `AIWORLD_ORIGINS=https://your-site` (add `AIWORLD_HOST=0.0.0.0` if it must be reachable from other machines). There is one world per server, so it is meant for a single viewer.

---

## 📜 The syllabus

Students move up through **seven levels**. To be promoted, a student must pass every subject of its level, meet the extra requirement (if any) and have built enough knowledge. Then it advances to the next level, and so on.

| Level | Name | Subjects | Also needed | Knowledge to leave |
|---|---|---|---|---|
| 1 | Foundations | Left vs Right · Top vs Bottom · Diagonal | | 2 notes |
| 2 | Shapes | Stripe · Circle · Diamond | | 4 notes, 📘 Learner |
| 3 | Curves & Logic | Ellipse · XOR · Waves | | 8 notes, 📘 Learner |
| 4 | Patterns | Rings · Checkerboard | pass a real-data subject | 12 notes, 🎓 Expert |
| 5 | Mastery | Flower · Spiral | | 16 notes, 🎓 Expert |
| 6 | Invention | | create a puzzle or invent a sense | 20 notes, 🏆 Master of a field |
| 7 | Open research | no gates | | |

The order follows curriculum learning (easy before hard), the dataset order of TensorFlow Playground, UNESCO's education levels and Bloom's taxonomy. Read the reasoning and the references in **[docs/SYLLABUS.md](docs/SYLLABUS.md)**.

---

## 🧪 Testing

```bash
python brain.py
# Runs: learning test · grow-neuron test · add-sense test
#       formula parser (accepts maths, rejects arbitrary code) · policy learning
```

No test framework yet — adding `pytest` is a great first contribution.

---

## 🤝 Contributing

All contributions welcome — from fixing a typo to adding a whole new kind of student. See **[CONTRIBUTING.md](CONTRIBUTING.md)** for setup, conventions and the review checklist.

**Good first issues**
- 🧪 Add `pytest` tests for `brain.py` (parser edge cases, grow/add-sense, policy learning)
- 💾 Persist student network weights so they survive a restart
- 🔎 Replace keyword retrieval with embeddings (`nomic-embed-text` via Ollama) and measure the difference
- 🍎 macOS / Linux launcher scripts (`start-world.sh`)
- ♿ Keyboard navigation and screen-reader labels for the dashboard

**Bigger ideas**
- 🖥️ Headless mode — run the simulation server-side so the world lives 24/7 with no browser
- 🔬 LoRA fine-tuning of Gemma on `export_dataset.py` output, then compare exam scores before/after
- 🧬 Richer task environments (grid worlds, simple games) beyond blue/red dots
- 📈 CSV logging + plots (does the Gemma mind really save lessons?)
- 🗣️ Student-to-student teaching and peer conversation

---

## 🛡️ Safety

- **Read-only internet.** Students make HTTPS GET requests only — to Wikipedia, USGS and Open-Meteo. They never post, sign up, send messages or buy anything. Nothing about you is sent anywhere.
- **Safe formula engine.** Invented formulas go through a parser that understands only numbers, variables (`x y r a pi`), arithmetic and a short allowlist of maths functions. `eval` is never called.
- **Local server only.** The server listens on `127.0.0.1`. Write requests are accepted only from the world's own page (Origin check).
- **No telemetry.** Not a byte of usage data leaves your machine.

---

## 🙏 Credits

Built with love by **[Andrew Savio](https://github.com/andrewsavio)**.

Standing on the shoulders of:
[three.js](https://threejs.org) · [PyTorch](https://pytorch.org) · [Ollama](https://ollama.com) · [Gemma](https://ai.google.dev/gemma) · [Wikipedia](https://wikipedia.org) · [USGS Earthquake Hazards](https://earthquake.usgs.gov) · [Open-Meteo](https://open-meteo.com)

**Gemma** is a model by Google, used under the [Gemma Terms of Use](models/GEMMA_TERMS.txt). Model weights are **not** included in this repository.

---

## 📄 License

Code: **[MIT](LICENSE)**
Gemma weights: governed by [Google's Gemma Terms of Use](models/GEMMA_TERMS.txt).

---

<div align="center">

*"The best way to understand intelligence is to watch it grow from nothing."*

**⭐ Star this repo if it made you think differently about AI.**

</div>
