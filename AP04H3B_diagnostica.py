#!/usr/bin/env python3
"""AP-04H3B: diagnosi dell'apprendimento che non produce differenze osservabili.

NON e' un esperimento H3. Richiede AP04H3_pilota.py invariato nella stessa cartella.
Per il motore reale richiede Crafter 1.8.3 con la patch documentata nel workflow.
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import AP04H3_pilota as old


class InstrumentedAgent(old.Agent):
    def __init__(self, alpha):
        super().__init__(alpha)
        self.track = False
        self.train_states = set()
        self.eval_events = []
        self.eval_requests = []
        self.eval_seed = None

    def select(self, state, step, rng, training):
        known = state in self.train_states
        row = self.q.get(state)
        top = max(row) if row else 0.0
        requested = super().select(state, step, rng, training)
        if self.track and not training:
            self.eval_events.append({
                'seed': self.eval_seed,
                'step': step,
                'known_train_state': int(known),
                'positive_q': int(top > 0),
                'above_gate': int(top > 0.22),
                'override': int(requested != step % self.n_actions),
            })
            self.eval_requests.append((self.eval_seed, step, requested))
        return requested


def run_agent(name, alpha, factory, training_seeds, probe_seeds, horizon):
    a = InstrumentedAgent(alpha)
    pre = old.measure(a, factory, probe_seeds, 'baseline')
    # Evaluation invokes .select() and thus creates zero-initialized q rows.
    # Clear them: only genuinely visited training states count as training coverage.
    a.q.clear()
    # No telemetry during training; after training capture visited states and q.
    for seed in training_seeds:
        old.episode(factory, a, seed, training=True, horizon=horizon, shock_at=horizon // 2)
    a.train_states = set(a.q)
    qrows = list(a.q.values())
    qmax = max((max(vals) for vals in qrows), default=0.0)
    positive_contexts = sum(any(v > 0 for v in vals) for vals in qrows)
    gated_contexts = sum(max(vals) > 0.22 for vals in qrows)

    a.eval_events.clear()
    a.eval_requests.clear()
    a.track = True
    records = []
    counts = defaultdict(int)
    for seed in probe_seeds:
        a.eval_seed = seed
        r = old.episode(factory, a, seed, training=False, horizon=horizon,
                        shock_at=horizon // 2)
        records.append(r)
        for ach in r['achievements']:
            counts[ach] += 1
    post = sorted(k for k, n in counts.items() if n >= old.PROBE_THRESHOLD)
    before = set(pre['repeatable'])
    after = set(post)
    events = a.eval_events
    size = len(events)
    def freq(k): return sum(x[k] for x in events) / size if size else 0.0
    summary = {
        'agent': name,
        'alpha': alpha,
        'training_updates': a.updates,
        'trained_states': len(a.train_states),
        'trained_states_with_positive_q': positive_contexts,
        'trained_states_above_0_22_gate': gated_contexts,
        'max_learned_q': round(qmax, 6),
        'heldout_steps': size,
        'heldout_fraction_states_seen_in_training': round(freq('known_train_state'), 6),
        'heldout_fraction_positive_q': round(freq('positive_q'), 6),
        'heldout_fraction_gate_open': round(freq('above_gate'), 6),
        'heldout_fraction_actions_override_cycle': round(freq('override'), 6),
        'pre_repeatable_ids': sorted(before),
        'post_repeatable_ids': sorted(after),
        'new_repeatable_ids': sorted(after-before),
        'lost_repeatable_ids': sorted(before-after),
        'net_change_repeatable': len(after)-len(before),
        'new_count': len(after-before),
        'lost_count': len(before-after),
        'heldout_mean_total_achievements': round(old.mean([r['total_achievements'] for r in records]), 6),
        'heldout_mean_health': round(old.mean([r['health_end'] for r in records]), 6),
    }
    return summary, a.eval_requests


def run(factory, out, train_count, probe_count, horizon):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    traces = {}
    for name, alpha in [('frozen', 0.0), ('slow', 0.12), ('fast', 0.5)]:
        info, requests = run_agent(name, alpha, factory,
                                   list(range(1000, 1000 + train_count)),
                                   list(range(2000, 2000 + probe_count)), horizon)
        results.append(info)
        traces[name] = dict(((s, t), act) for s, t, act in requests)
        print(f"{name}: train Q states={info['trained_states']} "
              f"test overrides={info['heldout_fraction_actions_override_cycle']:.3f} "
              f"new={info['new_count']} lost={info['lost_count']}")
    frozen = traces['frozen']
    for info in results:
        other = traces[info['agent']]
        joint = sorted(frozen.keys() & other.keys())
        info['paired_action_disagreement_vs_frozen'] = (
            round(sum(frozen[k] != other[k] for k in joint)/len(joint), 6) if joint else None
        )
        info['paired_action_steps'] = len(joint)
    report = {
        'type': 'AP04H3B_DIAGNOSTICS_NOT_H3_RESULT',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'real_engine': factory.__name__ == 'real_crafter_factory',
        'train_seeds': [1000, 1000+train_count-1],
        'probe_seeds': [2000, 2000+probe_count-1],
        'horizon': horizon,
        'gate_threshold': 0.22,
        'method': 'Frozen, slow, fast trained separately. Identical held-out worlds; record policy overrides, visited trained-state fraction, and actual capability ID changes.',
        'agents': results,
    }
    (out/'diagnostica.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf8')
    fields = [k for k in results[0] if not k.endswith('_ids')]
    with (out/'diagnostica.csv').open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k:r[k] for k in fields} for r in results])
    print('Diagnostica:', (out/'diagnostica.json').resolve())
    return report


def real_crafter_factory(seed, horizon):
    import crafter
    return crafter.Env(seed=seed, length=horizon)


def mock_factory(seed, horizon):
    return old.MockEnv(seed, horizon)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--output', default='risultati_h3b')
    p.add_argument('--train', type=int, default=32)
    p.add_argument('--probes', type=int, default=8)
    p.add_argument('--steps', type=int, default=80)
    a = p.parse_args()
    if a.self_test:
        r = run(mock_factory, a.output, 4, 4, 24)
        assert r['real_engine'] is False
        assert len(r['agents']) == 3
        assert r['agents'][0]['training_updates'] == 0
        assert r['agents'][1]['training_updates'] > 0
        assert r['agents'][2]['training_updates'] > 0
        assert all(0 <= x['heldout_fraction_actions_override_cycle'] <= 1 for x in r['agents'])
        print('SELF TEST MOCK OK; NON sono dati di Crafter')
    else:
        from importlib.metadata import version
        assert version('crafter') == '1.8.3'
        run(real_crafter_factory, a.output, a.train, a.probes, a.steps)


if __name__ == '__main__':
    main()
