"""The students' neural brains, in Python with PyTorch.

Two kinds of brain live here, one per student:
  * Brain  - the "skill brain": a small neural network (MLP) that learns to separate blue from red dots.
             It grows neurons, gains senses (input features) and is trained with backpropagation (PyTorch autograd).
  * Policy - the "decision brain": a small neural network that learns *what to do next* (study, sleep, explore...)
             from the rewards the world gives. It is trained with REINFORCE, a basic policy-gradient method.
The world (index.html) decides when things happen and asks this module over HTTP (see server.py).
Run `python brain.py` to self-test.
"""
import math
import random
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)  # these networks are tiny: extra threads only add overhead
LOCK = threading.Lock()   # the HTTP server is multi-threaded, the brains are not
BATCH = 32                # lessons per training step
LR_SCALE = 24             # the world's learning rates were made for 1-lesson steps; mean-loss batches need a bigger step (tuned on the real subjects)
GRID = 40

# ---------------------------------------------------------------- senses (input features)
SENSES = [  # same order as FEATURES in index.html
    ('x', lambda x, y: x), ('y', lambda x, y: y), ('x²', lambda x, y: x * x), ('y²', lambda x, y: y * y), ('x·y', lambda x, y: x * y),
    ('sin x', lambda x, y: torch.sin(3 * x)), ('sin y', lambda x, y: torch.sin(3 * y)),
    ('radius', lambda x, y: torch.hypot(x, y)), ('angle', lambda x, y: torch.atan2(y, x) / math.pi)]
N_BASE = len(SENSES)

# Invented senses arrive as formula text. This tiny parser only knows numbers, x y r a pi, + - * / % ^ ( ) and a short list
# of maths functions, so a formula can never run code. (Same grammar as the parser in index.html.)
FN = {'sin': torch.sin, 'cos': torch.cos, 'tan': torch.tan, 'abs': torch.abs, 'sqrt': lambda v: torch.sqrt(v.abs()), 'tanh': torch.tanh,
      'exp': lambda v: torch.exp(v.clamp(max=10)), 'floor': torch.floor, 'sign': torch.sign}


def _mod(p, q):
    return torch.where(q == 0, torch.zeros_like(p), ((p % q) + q) % q)


def _div(p, q):
    return torch.where(q == 0, torch.zeros_like(p), p / torch.where(q == 0, torch.ones_like(q), q))


def compile_formula(src):
    import re
    src = re.sub(r'\s+', '', str(src).lower()).replace('π', 'pi')
    toks = re.findall(r'\d*\.?\d+|[a-z]+|[-+*/%^(),]', src)
    if not src or ''.join(toks) != src:
        raise ValueError("it uses symbols I can't read")
    pos = 0

    def peek():
        return toks[pos] if pos < len(toks) else None

    def take():
        nonlocal pos
        pos += 1
        return toks[pos - 1] if pos <= len(toks) else None

    def expr():
        f = term()
        while peek() in ('+', '-'):
            op, l, r = take(), f, term()
            f = (lambda l, r: lambda v: l(v) + r(v))(l, r) if op == '+' else (lambda l, r: lambda v: l(v) - r(v))(l, r)
        return f

    def term():
        f = unary()
        while peek() in ('*', '/', '%'):
            op, l, r = take(), f, unary()
            f = {'*': lambda l, r: lambda v: l(v) * r(v), '/': lambda l, r: lambda v: _div(l(v), r(v)), '%': lambda l, r: lambda v: _mod(l(v), r(v))}[op](l, r)
        return f

    def unary():
        if peek() == '-':
            take()
            f = unary()
            return lambda v: -f(v)
        if peek() == '+':
            take()
        return power()

    def power():
        b = atom()
        if peek() != '^':
            return b
        take()
        e = unary()
        return lambda v: torch.pow(b(v), e(v))

    def atom():
        t = take()
        if t is None:
            raise ValueError('the formula ends too early')
        if t == '(':
            f = expr()
            if take() != ')':
                raise ValueError('expected ")"')
            return f
        if re.fullmatch(r'\d*\.?\d+', t):
            return lambda v, n=float(t): torch.full_like(v['x'], n)
        if t in ('x', 'y', 'r', 'a'):
            return lambda v, k=t: v[k]
        if t == 'pi':
            return lambda v: torch.full_like(v['x'], math.pi)
        if t in FN or t in ('min', 'max', 'mod'):
            if take() != '(':
                raise ValueError('expected "("')
            args = [expr()]
            while peek() == ',':
                take()
                args.append(expr())
            if take() != ')':
                raise ValueError('expected ")"')
            if t == 'mod':
                return lambda v: _mod(args[0](v), args[1](v))
            if t in ('min', 'max'):
                fn = torch.minimum if t == 'min' else torch.maximum
                def reduce(v):
                    out = args[0](v)
                    for g in args[1:]:
                        out = fn(out, g(v))
                    return out
                return reduce
            return lambda v: FN[t](args[0](v))
        raise ValueError(f'it uses the unknown word "{t}"')

    f = expr()
    if pos != len(toks):
        raise ValueError(f'unexpected "{toks[pos]}"')
    return f


class World:
    """Everything shared by all brains: the senses, the subjects (datasets) and cached feature columns."""

    def __init__(self):
        self.senses = list(SENSES)
        self.skills = {}   # id -> (train [N,3], test [M,3])
        self.cache = {}    # (skill id | 'grid', sense index, split) -> column
        centres = (torch.arange(GRID) + 0.5) / GRID * 2          # cell centres, 0..2
        self.grid = (centres.repeat(GRID) - 1, 1 - centres.repeat_interleave(GRID))  # row 0 is the top of the map (y = 1)

    def add_formula(self, name, expr, mean=0.0, sd=1.0):
        f = compile_formula(expr)  # raises ValueError for anything that isn't plain maths

        def sense(x, y):
            raw = f({'x': x, 'y': y, 'r': torch.hypot(x, y), 'a': torch.atan2(y, x) / math.pi})
            v = (torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0) - mean) / (sd or 1.0)
            return v.clamp(-3, 3)
        self.senses.append((name, sense))
        return len(self.senses) - 1

    def column(self, key, fi, xy):
        k = (key, fi)
        if k not in self.cache:
            self.cache[k] = self.senses[fi][1](xy[:, 0], xy[:, 1]).float()
        return self.cache[k]

    def features(self, key, feats, xy):
        return torch.stack([self.column(key, fi, xy) for fi in feats], dim=1)

    def set_skill(self, sid, train, test):
        self.skills[sid] = (torch.tensor(train, dtype=torch.float32).reshape(-1, 3), torch.tensor(test, dtype=torch.float32).reshape(-1, 3))
        for k in [k for k in self.cache if k[0] in ((sid, 'tr'), (sid, 'te'))]:
            del self.cache[k]


W = World()


class Brain:
    """One student's skill brain: input senses -> hidden tanh layers -> one sigmoid output (probability of BLUE)."""

    def __init__(self, hidden, lr):
        self.lr, self.feats, self.sizes = float(lr), [0, 1], [2, *hidden, 1]
        self.layers = [nn.Linear(a, b) for a, b in zip(self.sizes, self.sizes[1:])]
        for l in self.layers:
            bound = 1.5 / math.sqrt(l.in_features)
            nn.init.uniform_(l.weight, -bound, bound)
            nn.init.zeros_(l.bias)
        self.lessons = 0

    def params(self):
        return [p for l in self.layers for p in (l.weight, l.bias)]

    def forward(self, x, keep=False):
        acts = [x]
        for l in self.layers[:-1]:
            x = torch.tanh(l(x))
            acts.append(x)
        x = torch.sigmoid(self.layers[-1](x))
        acts.append(x)
        return acts if keep else x.squeeze(-1)

    def add_sense(self, fi):
        self.feats.append(fi)
        first = self.layers[0]
        extra = (torch.rand(first.out_features, 1) * 2 - 1) * 0.1
        first.weight = nn.Parameter(torch.cat([first.weight.data, extra], dim=1))
        first.in_features += 1
        self.sizes[0] += 1

    def grow(self, h):  # add one neuron to hidden layer h (an index into sizes)
        before, after = self.layers[h - 1], self.layers[h]
        bound = 1.5 / math.sqrt(before.in_features) * 0.5
        before.weight = nn.Parameter(torch.cat([before.weight.data, (torch.rand(1, before.in_features) * 2 - 1) * bound]))
        before.bias = nn.Parameter(torch.cat([before.bias.data, torch.zeros(1)]))
        before.out_features += 1
        after.weight = nn.Parameter(torch.cat([after.weight.data, (torch.rand(after.out_features, 1) * 2 - 1) * 0.1], dim=1))
        after.in_features += 1
        self.sizes[h] += 1

    def study(self, sid, steps, conf):
        """Train on `steps` random lessons of subject `sid` with backpropagation; return the new practice score."""
        train = W.skills[sid][0]
        X, y = W.features((sid, 'tr'), self.feats, train[:, :2]), train[:, 2]
        last = None
        for _ in range(max(1, math.ceil(steps / BATCH))):
            idx = torch.randint(len(y), (BATCH,))
            out = self.forward(X[idx])
            loss = F.binary_cross_entropy(out, y[idx])          # how wrong were the guesses?
            for p in self.params():
                p.grad = None
            loss.backward()                                       # autograd: how much did each connection contribute?
            with torch.no_grad():
                for p in self.params():
                    p -= self.lr * LR_SCALE * p.grad              # nudge every connection against its share of the error
            for o, t in zip(out.detach().tolist(), y[idx].tolist()):
                conf += (float((o > 0.5) == (t > 0.5)) - conf) * 0.002
            last = (idx[-1].item(), out[-1].item())
            self.lessons += BATCH
        i, o = last
        return {'conf': conf, 'lesson': {'x': train[i, 0].item(), 'y': train[i, 1].item(), 't': train[i, 2].item(), 'out': o}, 'steps': steps}

    @torch.no_grad()
    def exam(self, sid):
        test = W.skills[sid][1]
        out = self.forward(W.features((sid, 'te'), self.feats, test[:, :2]))
        return ((out > 0.5) == (test[:, 2] > 0.5)).float().mean().item()

    @torch.no_grad()
    def view(self, sid, px, py):
        """Everything the dashboard needs to draw this brain: weights, neuron activity at a probe point, and its map of guesses."""
        probe = torch.stack([W.senses[fi][1](torch.tensor([px]), torch.tensor([py])).float()[0] for fi in self.feats]).unsqueeze(0)
        acts = [a[0].tolist() for a in self.forward(probe, keep=True)]
        gx, gy = W.grid
        grid = self.forward(W.features(('grid',), self.feats, torch.stack([gx, gy], dim=1)))
        return {'sizes': self.sizes, 'feats': self.feats, 'W': [l.weight.tolist() for l in self.layers], 'acts': acts,
                'grid': [round(v, 3) for v in grid.tolist()], 'lessons': self.lessons, 'params': sum(p.numel() for p in self.params())}

    def save(self, path):
        """Save this brain's weights, senses, and architecture state to disk."""
        state = {
            'sizes': self.sizes,
            'feats': self.feats,
            'lr': self.lr,
            'lessons': self.lessons,
            'state_dict': [l.state_dict() for l in self.layers],
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path):
        """Reconstruct a brain from saved weights and architecture state."""
        state = torch.load(path, weights_only=True)
        hidden = state['sizes'][1:-1]
        brain = cls(hidden=hidden, lr=state['lr'])
        brain.sizes = list(state['sizes'])
        brain.feats = list(state['feats'])
        brain.lessons = int(state['lessons'])
        brain.layers = [nn.Linear(a, b) for a, b in zip(brain.sizes, brain.sizes[1:])]
        for layer, sd in zip(brain.layers, state['state_dict']):
            layer.load_state_dict(sd)
        return brain


ACTIONS = ['study', 'sleep', 'play', 'chat', 'read', 'exam', 'wander', 'browse', 'create']
N_STATE = 14


class Policy:
    """One student's decision brain. Learns a *correction* to the world's built-in instincts from the rewards it receives."""

    def __init__(self):
        self.net = nn.Sequential(nn.Linear(N_STATE, 16), nn.Tanh(), nn.Linear(16, len(ACTIONS)))
        nn.init.zeros_(self.net[2].weight)  # starts as pure instinct; experience gradually changes it
        nn.init.zeros_(self.net[2].bias)
        self.pending, self.baseline, self.updates, self.mean_reward = {}, 0.0, 0, 0.0

    def _logits(self, state, prior, mask):
        return (prior + self.net(state)).masked_fill(~mask, float('-inf'))

    def act(self, state, prior):
        mask = torch.tensor([a in prior for a in ACTIONS])
        pr = torch.tensor([prior.get(a, 0.0) for a in ACTIONS])
        st = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            probs = F.softmax(self._logits(st, pr, mask), dim=0)
            resid = self.net(st).tolist()
        i = torch.multinomial(probs, 1).item()
        self.pending[ACTIONS[i]] = (st, pr, mask, i)
        return {'act': ACTIONS[i], 'probs': {a: round(p, 4) for a, p in zip(ACTIONS, probs.tolist()) if a in prior},
                'resid': {a: round(r, 3) for a, r in zip(ACTIONS, resid) if a in prior}}

    def reward(self, act, r):
        """REINFORCE: make actions that earned more than usual more likely, and those that earned less, less likely."""
        if act not in self.pending:
            return
        st, pr, mask, i = self.pending.pop(act)
        r = max(-2.0, min(2.0, float(r)))
        adv, self.baseline = r - self.baseline, self.baseline + 0.1 * (r - self.baseline)
        logp = F.log_softmax(self._logits(st, pr, mask), dim=0)
        probs = logp.exp()
        entropy = -(probs * logp.masked_fill(~mask, 0.0)).sum()
        loss = -adv * logp[i] - 0.01 * entropy
        for p in self.net.parameters():
            p.grad = None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        with torch.no_grad():
            for p in self.net.parameters():
                p -= 0.05 * p.grad
        self.updates += 1
        self.mean_reward += 0.05 * (r - self.mean_reward)

    def save(self, path):
        """Save policy network weights and training statistics to disk."""
        state = {
            'state_dict': self.net.state_dict(),
            'baseline': self.baseline,
            'updates': self.updates,
            'mean_reward': self.mean_reward,
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path):
        """Reconstruct a policy from saved weights and training statistics."""
        state = torch.load(path, weights_only=True)
        policy = cls()
        policy.net.load_state_dict(state['state_dict'])
        policy.baseline = float(state.get('baseline', 0.0))
        policy.updates = int(state.get('updates', 0))
        policy.mean_reward = float(state.get('mean_reward', 0.0))
        return policy


BRAINS, POLICIES = {}, {}


def handle(path, data):
    """One entry point for every call from the world. Returns a JSON-able dict."""
    global W
    with LOCK:
        if path == '/brain/reset':
            W = World()
            BRAINS.clear(), POLICIES.clear()
            for a in data['agents']:
                BRAINS[a['id']], POLICIES[a['id']] = Brain(a['hidden'], a['lr']), Policy()
            return {'ok': True}
        if path == '/brain/skills':
            for s in data['skills']:
                W.set_skill(s['id'], s['train'], s['test'])
            return {'ok': True}
        if path == '/brain/sense':
            return {'index': W.add_formula(str(data['name'])[:30], data['expr'], float(data.get('mean', 0.0)), float(data.get('sd', 1.0)))}
        if path == '/brain/add_sense':
            BRAINS[data['id']].add_sense(int(data['fi']))
            return {'ok': True}
        if path == '/brain/grow':
            BRAINS[data['id']].grow(int(data['layer']))
            return {'ok': True}
        if path == '/brain/save':
            aid = data['id']
            BRAINS[aid].save(data['path'])
            if 'policy_path' in data and aid in POLICIES:
                POLICIES[aid].save(data['policy_path'])
            return {'ok': True}
        if path == '/brain/load':
            aid = data['id']
            BRAINS[aid] = Brain.load(data['path'])
            if 'policy_path' in data:
                POLICIES[aid] = Policy.load(data['policy_path'])
            return {'ok': True}
        if path == '/brain/study':
            return {'results': [{'id': j['id'], **BRAINS[j['id']].study(j['skill'], int(j['steps']), float(j['conf']))} for j in data['jobs']]}
        if path == '/brain/exam':
            return {'acc': BRAINS[data['id']].exam(data['skill'])}
        if path == '/brain/view':
            return BRAINS[data['id']].view(data['skill'], float(data['px']), float(data['py']))
        if path == '/policy/act':
            return POLICIES[data['id']].act(data['state'], data['prior'])
        if path == '/policy/reward':
            POLICIES[data['id']].reward(data['act'], data['reward'])
            return {'ok': True}
        if path == '/brain/info':
            return {'engine': f'PyTorch {torch.__version__}', 'device': 'cpu', 'brains': len(BRAINS),
                    'params': sum(sum(p.numel() for p in b.params()) for b in BRAINS.values()),
                    'lessons': sum(b.lessons for b in BRAINS.values()), 'policy_updates': sum(p.updates for p in POLICIES.values()),
                    'policy': {i: {'updates': p.updates, 'reward': round(p.mean_reward, 3)} for i, p in POLICIES.items()}}
    raise KeyError(path)


# ---------------------------------------------------------------- self-test: python brain.py
def _selftest():
    import time
    random.seed(1), torch.manual_seed(1)
    # 1. a brain learns a rule it was never told (circle: blue inside radius 0.63), and is judged on unseen dots
    def dots(n, f):
        pts = [(random.uniform(-1, 1), random.uniform(-1, 1)) for _ in range(n)]
        return [[x, y, float(f(x, y))] for x, y in pts]
    rule = lambda x, y: x * x + y * y < 0.4
    handle('/brain/reset', {'agents': [{'id': 0, 'hidden': [6, 4], 'lr': 0.05}]})
    handle('/brain/skills', {'skills': [{'id': 0, 'train': dots(600, rule), 'test': dots(300, rule)}]})
    before = handle('/brain/exam', {'id': 0, 'skill': 0})['acc']
    handle('/brain/add_sense', {'id': 0, 'fi': 7})  # the "radius" sense makes a circle easy
    t = time.perf_counter()
    conf = 0.5
    for _ in range(40):
        conf = handle('/brain/study', {'jobs': [{'id': 0, 'skill': 0, 'steps': 150, 'conf': conf}]})['results'][0]['conf']
    after = handle('/brain/exam', {'id': 0, 'skill': 0})['acc']
    print(f'circle: exam {before:.0%} -> {after:.0%} after 6000 lessons ({time.perf_counter() - t:.1f}s)')
    assert after > 0.93, 'the brain did not learn the circle'
    # 2. growing a neuron and adding a sense keep the brain working
    handle('/brain/grow', {'id': 0, 'layer': 1})
    v = handle('/brain/view', {'id': 0, 'skill': 0, 'px': 0.1, 'py': 0.2})
    assert v['sizes'] == [3, 7, 4, 1] and len(v['grid']) == GRID * GRID and len(v['W'][0][0]) == 3
    assert handle('/brain/exam', {'id': 0, 'skill': 0})['acc'] > 0.8, 'growing a neuron broke the brain'
    # 3. the formula parser: right answers, and it refuses anything that is not maths
    x, y = torch.tensor([0.5]), torch.tensor([-0.25])
    v = {'x': x, 'y': y, 'r': torch.hypot(x, y), 'a': torch.atan2(y, x) / math.pi}
    for src, want in [('x^2 - y^2', 0.1875), ('-x^2', -0.25), ('2^3^2', 512.0), ('3*-x', -1.5), ('max(x, y, 0.1)', 0.5), ('x/0', 0.0), ('mod(7, 3)', 1.0), ('sin(9*r)', math.sin(9 * v['r'].item()))]:
        assert abs(compile_formula(src)(v).item() - want) < 1e-5, src
    for bad in ['__import__("os")', 'x;y', '2x', 'open(1)', 'sin(', '']:
        try:
            compile_formula(bad)
            raise AssertionError(f'accepted {bad!r}')
        except ValueError:
            pass
    # 4. the decision brain learns from reward: rewarding "play" and punishing "study" shifts its choices
    p = POLICIES[0]
    state, prior = [0.5] * N_STATE, {'study': 0.0, 'play': 0.0, 'sleep': 0.0}
    p0 = p.act(state, prior)['probs']['play']
    for _ in range(300):
        a = handle('/policy/act', {'id': 0, 'state': state, 'prior': prior})['act']
        handle('/policy/reward', {'id': 0, 'act': a, 'reward': 1.0 if a == 'play' else -1.0})
    p1 = p.act(state, prior)['probs']['play']
    print(f'policy: chance of choosing play {p0:.0%} -> {p1:.0%} after 300 rewarded decisions')
    assert p1 > p0 + 0.2, 'the decision brain did not learn from reward'
    # 5. persistence: saving and loading preserves weights and predictions
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        bpath = f'{tmpdir}/brain.pt'
        ppath = f'{tmpdir}/policy.pt'
        handle('/brain/save', {'id': 0, 'path': bpath, 'policy_path': ppath})
        b_loaded = Brain.load(bpath)
        p_loaded = Policy.load(ppath)
        assert b_loaded.sizes == BRAINS[0].sizes and b_loaded.lessons == BRAINS[0].lessons
        assert p_loaded.updates == POLICIES[0].updates
        probe = torch.tensor([[0.2, -0.3, 0.5]])
        assert torch.allclose(BRAINS[0].forward(probe), b_loaded.forward(probe))
        assert torch.allclose(POLICIES[0].net(torch.tensor(state)), p_loaded.net(torch.tensor(state)))
    print('brain self-test passed')


if __name__ == '__main__':
    _selftest()
