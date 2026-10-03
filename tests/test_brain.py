"""Unit and integration tests for AI World neural brains, policy gradients, parser, and persistence."""
import math
import random
import pytest
import torch

import brain
from brain import Brain, Policy, World, compile_formula, handle, ACTIONS, N_STATE, GRID


@pytest.fixture(autouse=True)
def set_random_seeds():
    random.seed(42)
    torch.manual_seed(42)


def generate_dots(n, rule):
    pts = [(random.uniform(-1, 1), random.uniform(-1, 1)) for _ in range(n)]
    return [[x, y, float(rule(x, y))] for x, y in pts]


def test_circle_learning():
    """Verify an MLP can learn a non-linear circular decision boundary via backprop."""
    rule = lambda x, y: x * x + y * y < 0.4
    train_data = generate_dots(600, rule)
    test_data = generate_dots(300, rule)

    handle('/brain/reset', {'agents': [{'id': 0, 'hidden': [6, 4], 'lr': 0.05}]})
    handle('/brain/skills', {'skills': [{'id': 0, 'train': train_data, 'test': test_data}]})

    before = handle('/brain/exam', {'id': 0, 'skill': 0})['acc']
    handle('/brain/add_sense', {'id': 0, 'fi': 7})  # add radius sense

    conf = 0.5
    for _ in range(40):
        conf = handle('/brain/study', {'jobs': [{'id': 0, 'skill': 0, 'steps': 150, 'conf': conf}]})['results'][0]['conf']

    after = handle('/brain/exam', {'id': 0, 'skill': 0})['acc']
    assert after > 0.93, f'Accuracy {after:.2%} did not reach threshold of 93%'


def test_grow_neuron_and_add_sense():
    """Verify dynamic network resizing (adding senses and growing hidden neurons) preserves functionality."""
    rule = lambda x, y: x * x + y * y < 0.4
    train_data = generate_dots(300, rule)
    test_data = generate_dots(150, rule)

    handle('/brain/reset', {'agents': [{'id': 0, 'hidden': [6, 4], 'lr': 0.05}]})
    handle('/brain/skills', {'skills': [{'id': 0, 'train': train_data, 'test': test_data}]})
    handle('/brain/add_sense', {'id': 0, 'fi': 7})

    for _ in range(20):
        handle('/brain/study', {'jobs': [{'id': 0, 'skill': 0, 'steps': 150, 'conf': 0.5}]})

    # Grow a neuron in hidden layer 1 (index 1)
    handle('/brain/grow', {'id': 0, 'layer': 1})
    v = handle('/brain/view', {'id': 0, 'skill': 0, 'px': 0.1, 'py': 0.2})

    assert v['sizes'] == [3, 7, 4, 1]
    assert len(v['grid']) == GRID * GRID
    assert len(v['W'][0][0]) == 3
    exam_acc = handle('/brain/exam', {'id': 0, 'skill': 0})['acc']
    assert exam_acc > 0.75, f'Exam accuracy {exam_acc:.2%} degraded after growing neuron'


def test_policy_reinforce_learning():
    """Verify REINFORCE policy gradients shift action selection towards rewarded behavior."""
    policy = Policy()
    state = [0.5] * N_STATE
    prior = {'study': 0.0, 'play': 0.0, 'sleep': 0.0}

    initial_play_prob = policy.act(state, prior)['probs']['play']

    for _ in range(300):
        decision = policy.act(state, prior)
        chosen = decision['act']
        reward = 1.0 if chosen == 'play' else -1.0
        policy.reward(chosen, reward)

    final_play_prob = policy.act(state, prior)['probs']['play']
    assert final_play_prob > initial_play_prob + 0.25, (
        f'Policy failed to learn: initial {initial_play_prob:.2f} -> final {final_play_prob:.2f}'
    )
    assert policy.updates == 300


@pytest.mark.parametrize('formula,expected', [
    ('x^2 - y^2', 0.1875),
    ('-x^2', -0.25),
    ('2^3^2', 512.0),
    ('3*-x', -1.5),
    ('max(x, y, 0.1)', 0.5),
    ('min(x, y)', -0.25),
    ('x/0', 0.0),
    ('mod(7, 3)', 1.0),
    ('sin(9*r)', math.sin(9 * math.hypot(0.5, -0.25))),
    ('exp(1)', math.e),
    ('abs(y)', 0.25),
    ('sign(y)', -1.0),
    ('floor(2.8)', 2.0),
])
def test_formula_parser_valid(formula, expected):
    """Verify recursive descent formula parser correctly computes mathematical expressions."""
    x, y = torch.tensor([0.5]), torch.tensor([-0.25])
    v = {'x': x, 'y': y, 'r': torch.hypot(x, y), 'a': torch.atan2(y, x) / math.pi}
    fn = compile_formula(formula)
    result = fn(v).item()
    assert abs(result - expected) < 1e-4, f'{formula}: got {result}, expected {expected}'


@pytest.mark.parametrize('bad_expression', [
    '__import__("os")',
    'x;y',
    '2x',
    'open("brain.py")',
    'sin(',
    '',
    'x +',
    'eval("1+1")',
    'system("ls")',
    '(x + y',
    'foo(x)',
])
def test_formula_parser_rejects_unsafe_and_invalid(bad_expression):
    """Verify formula parser rejects malicious code injection and syntax errors."""
    with pytest.raises(ValueError):
        compile_formula(bad_expression)


def test_world_add_formula():
    """Verify World safely accepts compiled formula senses and computes feature activations."""
    world = World()
    idx = world.add_formula('test_hypot', 'sqrt(x^2 + y^2)', mean=0.0, sd=1.0)
    assert idx >= len(brain.SENSES)

    xy = torch.tensor([[0.3, 0.4]])
    col = world.column(('test',), idx, xy)
    assert abs(col[0].item() - 0.5) < 1e-5


def test_brain_persistence(tmp_path):
    """Verify saving and loading Brain preserves architecture, weights, and forward pass results."""
    b = Brain(hidden=[8, 6], lr=0.02)
    b.add_sense(5)
    b.grow(1)
    b.lessons = 1200

    probe = torch.randn(5, b.sizes[0])
    orig_output = b.forward(probe)

    save_path = tmp_path / 'brain_model.pt'
    b.save(str(save_path))

    loaded_b = Brain.load(str(save_path))
    assert loaded_b.sizes == b.sizes
    assert loaded_b.feats == b.feats
    assert loaded_b.lessons == 1200
    assert loaded_b.lr == 0.02

    loaded_output = loaded_b.forward(probe)
    assert torch.allclose(orig_output, loaded_output, atol=1e-6)


def test_policy_persistence(tmp_path):
    """Verify saving and loading Policy preserves neural network weights and statistics."""
    p = Policy()
    state = [0.2] * N_STATE
    prior = {'study': 0.0, 'play': 0.0}

    # Generate some updates
    for _ in range(10):
        decision = p.act(state, prior)
        p.reward(decision['act'], 1.0)

    save_path = tmp_path / 'policy_model.pt'
    p.save(str(save_path))

    loaded_p = Policy.load(str(save_path))
    assert loaded_p.updates == p.updates
    assert abs(loaded_p.baseline - p.baseline) < 1e-6
    assert abs(loaded_p.mean_reward - p.mean_reward) < 1e-6

    st_tensor = torch.tensor(state, dtype=torch.float32)
    assert torch.allclose(p.net(st_tensor), loaded_p.net(st_tensor))


def test_handle_persistence_api(tmp_path):
    """Verify /brain/save and /brain/load work through the handle dispatcher."""
    handle('/brain/reset', {'agents': [{'id': 0, 'hidden': [4], 'lr': 0.05}]})
    b_file = str(tmp_path / 'b0.pt')
    p_file = str(tmp_path / 'p0.pt')

    save_res = handle('/brain/save', {'id': 0, 'path': b_file, 'policy_path': p_file})
    assert save_res['ok'] is True

    # Modify the in-memory brain lessons
    brain.BRAINS[0].lessons = 9999

    load_res = handle('/brain/load', {'id': 0, 'path': b_file, 'policy_path': p_file})
    assert load_res['ok'] is True
    assert brain.BRAINS[0].lessons == 0  # Restored to 0
