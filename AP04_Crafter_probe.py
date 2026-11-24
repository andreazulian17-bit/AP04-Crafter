#!/usr/bin/env python3
"""AP-04 / Crafter: suitability smoke test, NOT a test of H3.

Usage:
  python AP-04_Crafter_probe.py --self-test
  pip install crafter==1.8.3
  python AP-04_Crafter_probe.py --seed 42 --steps 80 --shock-at 12

No model is trained in this test. Perturbations act on the action interface
(not directly on environmental physics). Output is a JSON audit trail.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if hasattr(value, 'item'):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def digest(value):
    if hasattr(value, 'tobytes'):
        raw = value.tobytes()
        shape = str(getattr(value, 'shape', ''))
        raw += shape.encode('utf8')
    else:
        raw = json.dumps(_jsonable(value), sort_keys=True).encode('utf8')
    return hashlib.sha256(raw).hexdigest()[:20]


def actions_for_steps(n, count):
    # Reproducible diagnostic sequence (not an intelligent agent).
    # Always includes no-op and a broad sample of actions.
    return [i % n for i in range(count)]


def unpack_reset(value):
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], dict):
        return value[0]
    return value


def unpack_step(value):
    if len(value) == 4:
        obs, reward, done, info = value
    elif len(value) == 5:
        obs, reward, terminated, truncated, info = value
        done = terminated or truncated
    else:
        raise RuntimeError(f'Unexpected step signature ({len(value)} elements)')
    return obs, float(reward), bool(done), info


def execute(factory, seed, steps, shock_at, condition):
    env = factory(seed, steps)
    try:
        obs = unpack_reset(env.reset())
        n_actions = int(env.action_space.n)
        if n_actions < 3:
            raise RuntimeError('Expected >= 3 actions for action remapping')
        scripted = actions_for_steps(n_actions, steps)
        rows = []
        start_digest = digest(obs)
        for index, intended in enumerate(scripted):
            actual = intended
            if index >= shock_at:
                if condition == 'remap':
                    actual = (intended + 1) % n_actions
                elif condition == 'lockout':
                    actual = 0
            obs, reward, done, info = unpack_step(env.step(actual))
            info = info or {}
            achievements = _jsonable(info.get('achievements', {}))
            inventory = _jsonable(info.get('inventory', {}))
            row = {
                'step': index + 1,
                'requested_action': intended,
                'executed_action': actual,
                'reward': reward,
                'done': done,
                'observation_digest': digest(obs),
                'achievements': achievements,
                'inventory': inventory,
            }
            rows.append(row)
            if done:
                break
        last_achievements = rows[-1]['achievements'] if rows else {}
        return {
            'condition': condition, 'seed': seed, 'action_count': n_actions,
            'initial_observation_digest': start_digest,
            'steps_executed': len(rows),
            'reached_intervention_step': len(rows) > shock_at,
            'intervention_executed': any(
                r['requested_action'] != r['executed_action'] for r in rows),
            'terminal': rows[-1]['done'] if rows else False,
            'total_reward': sum(x['reward'] for x in rows),
            'achievements_unlocked': sum(1 for v in last_achievements.values() if v)
                if isinstance(last_achievements, dict) else None,
            'rows': rows,
        }
    finally:
        if hasattr(env, 'close'):
            env.close()


def compact_signature(record):
    # identical runs must produce the same states, rewards, inventory and achievements.
    # Exclude descriptive metadata.
    return digest(record['rows'])


def compare(a, b):
    for idx, (x, y) in enumerate(zip(a['rows'], b['rows'])):
        if x != y:
            return idx + 1
    if len(a['rows']) != len(b['rows']):
        return min(len(a['rows']), len(b['rows'])) + 1
    return None


def do_probe(factory, name, seed, steps, shock_at):
    runs = [execute(factory, seed, steps, shock_at, case)
            for case in ['baseline', 'baseline', 'remap', 'lockout']]
    base1, base2, remap, lockout = runs
    deterministic = (compact_signature(base1) == compact_signature(base2))
    result = {
        'test_kind': 'ENGINE_SUITABILITY_SMOKE_TEST_NOT_H3',
        'tested_engine': name,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'python_version': platform.python_version(),
        'seed': seed, 'planned_steps': steps, 'shock_at_step_0_based': shock_at,
        'engine_reproducibility_same_seed_same_actions': deterministic,
        'action_interface_interventions': {
            'remap_first_difference_from_baseline_step': compare(base1, remap),
            'lockout_first_difference_from_baseline_step': compare(base1, lockout),
            'remap_actually_changed_action': remap['intervention_executed'],
            'lockout_actually_changed_action': lockout['intervention_executed'],
        },
        'conditions': runs,
        'conclusions_not_tested': [
            'No agent learning or new capability acquisition',
            'No direct modification of environment physical rules',
            'No proof of KP or Gamma effects on survival',
            'An achievement is not automatically a repeatable capacity',
        ],
    }
    return result


class MockActionSpace:
    n = 17


class MockEnv:
    """Test double checks probe logic only; NOT the Crafter engine."""
    action_space = MockActionSpace()

    def __init__(self, seed, length):
        self.state = seed % 23
        self.length = length
        self.t = 0

    def reset(self):
        self.t = 0
        return [self.state, self.t]

    def step(self, action):
        self.t += 1
        self.state = (self.state * 3 + int(action) * 7 + 1) % 97
        info = {
            'achievements': {'first': int(self.t >= 3), 'rare': int(self.state == 42)},
            'inventory': {'wood': self.state % 3},
        }
        return [self.state, self.t], float(action == 1), self.t >= self.length, info


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test', action='store_true', help='test this harness with a mock, not Crafter')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--steps', type=int, default=80)
    p.add_argument('--shock-at', type=int, default=12)
    p.add_argument('--output', default='AP-04_Crafter_probe_results.json')
    args = p.parse_args()
    if not (2 <= args.shock_at < args.steps):
        p.error('Need 2 <= --shock-at < --steps')

    if args.self_test:
        factory = lambda seed, steps: MockEnv(seed, steps)
        engine_name = 'MockEnv (probe logic only; not Crafter)'
        provenance = {'kind': 'mock', 'version': None, 'module_file': None}
    else:
        try:
            import crafter
        except Exception as exc:
            print(f'BLOCKED: Crafter could not be imported ({exc}).', file=sys.stderr)
            print('Install in a supported Python environment with: pip install crafter==1.8.3',
                  file=sys.stderr)
            return 2
        factory = lambda seed, steps: crafter.Env(seed=seed, length=steps)
        from importlib.metadata import version as distribution_version
        installed_version = distribution_version('crafter')
        engine_name = f'crafter {installed_version}'
        provenance = {'kind': 'installed-crafter', 'version': installed_version, 'module_file': str(getattr(crafter, '__file__', ''))}

    report = do_probe(factory, engine_name, args.seed, args.steps, args.shock_at)
    report['engine_provenance'] = provenance
    out = Path(args.output)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf8')
    short = {
        k: v for k, v in report.items() if k != 'conditions'
    }
    short['run_summaries'] = [{k: r[k] for k in (
        'condition', 'steps_executed', 'terminal', 'achievements_unlocked',
        'reached_intervention_step', 'total_reward')} for r in report['conditions']]
    print(json.dumps(short, indent=2, ensure_ascii=False))
    print(f'Full audit: {out.resolve()}')
    if not report['engine_reproducibility_same_seed_same_actions']:
        return 1
    if args.self_test:
        assert report['action_interface_interventions']['remap_actually_changed_action']
        assert report['action_interface_interventions']['lockout_actually_changed_action']
        assert report['action_interface_interventions']['remap_first_difference_from_baseline_step'] is not None
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
