# Contributing to AI World

Thanks for helping! This project is meant to be **readable and hackable**: a good place to learn how neural networks, reinforcement learning, retrieval and a 3D front end fit together.

## Set up

```bash
git clone https://github.com/andrewsavio/AI-World.git
cd AI-World
pip install -r requirements.txt     # torch, pystray, pillow
python brain.py                     # self-test, should print "brain self-test passed"
python server.py                    # http://localhost:8000
```

For the professor and student minds, install [Ollama](https://ollama.com) and run `ollama pull gemma2:2b && ollama cp gemma2:2b aiworld-gemma`. Everything except those features works without it.

## How the code is organised

- **`index.html`** holds the simulation and the 3D world. Search for the section banners (`// =====`) to jump around. State lives in plain objects (`agents`, `SKILLS`, `FEATURES`); there is no build step and no framework.
- **`brain.py`** is the only place with PyTorch. The page talks to it through `handle(path, data)`; `server.py` just exposes that over HTTP.
- The page has no brain of its own: all learning goes through `brain.py`. If Python is unreachable the world pauses and reconnects by itself; the page remembers each brain's shape (sizes, senses) so a restarted Python can be rebuilt (`pySyncAll`).

## Ground rules

1. **Keep it light.** The target is integrated graphics. Prefer instancing and merged meshes, avoid per-frame allocations, and test with `Quality: Low`.
2. **Keep the students' internet read-only.** No posting, accounts, purchases or anything that sends user data.
3. **Treat model output and web text as untrusted.** Render it with `esc()` (plain text), never as HTML, and never `eval` it. New formula functions go in `FN` in *both* `brain.py` and `index.html`.
4. **Explain the ML.** This is a learning project. Short comments that say *why* (not what) are valued, and the in-app guide (`#guide`) should stay accurate.
5. **Small, focused PRs** with a clear description and, for visible changes, a screenshot.

## Before you open a pull request

- [ ] `python brain.py` passes
- [ ] The page loads with no console errors, with and without Ollama running
- [ ] Syntax check the page script if you edited it (`node --check` on the contents of the `<script type="module">` block works)
- [ ] You did not commit model weights, `students-memory*.json`, `professor-memory.json` or `training-data.jsonl` (they are git-ignored)

## Commit messages

Short imperative subject ("Add embedding retrieval"), then a few lines on *why*.

## Ideas

See the **Good first issues** and **Bigger ideas** lists in the README, or open an issue to discuss something new first.
