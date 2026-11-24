#!/usr/bin/env python3
"""AP-02: capacita' di riconfigurazione dopo un guasto, con scadenza.

Toy model completamente deterministico, solo libreria standard Python.
Esecuzione:
    python AP-02_riconfigurazione.py
    python AP-02_riconfigurazione.py --details
    python AP-02_riconfigurazione.py --max-steps 3

Sistemi: grafi diretti a 6 stati e 6 archi, uguali ad AP-01.
Missione: stato iniziale e finale scelti fra i 30 abbinamenti ordinati,
          equiprobabili. Una missione riesce entro K transizioni.
Perturbazione: viene cancellato uno dei 6 archi originali, equiprobabile.

Politica 0: nessuna riconfigurazione.
Politica 1: dopo aver osservato il guasto (ma PRIMA di conoscere la missione),
            il sistema puo' costruire al piu' un nuovo collegamento fisico
            diretto, di costo unitario. Sono candidabili soltanto coppie (u,v)
            gia' collegate da un cammino superstite di ESATTAMENTE due archi.
            Vietato riattivare esattamente l'arco guasto. Sceglie il candidato
            che massimizza il numero di missioni eseguibili entro K passi;
            in caso di parita', ordine alfabetico. Non costruisce nulla se
            il guadagno e' zero.

ATTENZIONE FISICA:
- "Costruire un collegamento diretto" e' un'ASSUNZIONE del giocattolo, non una
  conseguenza della sola composizione. Il costo di costruzione e' contabilizzato
  separatamente e non incluso nel deadline K della missione successiva.
- Senza questa nuova operazione fisica indipendente, una macro che concatena
  i due archi superstiti impiega ancora due passi: non offre guadagno fisico.
- Un collegamento gia' raggiungibile non cambia la raggiungibilita' illimitata;
  puo' pero' migliorare quella entro una scadenza, se e solo se davvero puo'
  essere eseguito come una singola operazione fisica.
- Il sistema usa un ottimizzatore esaustivo (oracolo del grafo), NON apprende,
  non inventa un'attivita' non prevista dal repertorio delle trasformazioni.
- Le misure sono relative a questa classe prefissata di sei guasti e trenta
  missioni; non rappresentano la persistenza contro qualsiasi imprevisto.
"""

import argparse
from collections import deque
from itertools import product

STATES = tuple('ABCDEF')
TASKS = {(a, b) for a, b in product(STATES, repeat=2) if a != b}
SYSTEMS = {
    'S1 — Specialista': [
        ('A', 'B'), ('B', 'A'), ('A', 'C'),
        ('C', 'A'), ('B', 'C'), ('C', 'B'),
    ],
    'S2 — Compositore': [
        ('A', 'B'), ('B', 'D'), ('A', 'C'),
        ('C', 'D'), ('D', 'E'), ('E', 'F'),
    ],
    'S3 — Esploratore': [
        ('A', 'B'), ('B', 'C'), ('C', 'D'),
        ('D', 'E'), ('E', 'F'), ('F', 'A'),
    ],
}


def reach(edges, max_steps=None):
    """Coppie direzionali raggiungibili con al piu' max_steps archi."""
    adj = {s: [] for s in STATES}
    for u, v in edges:
        adj[u].append(v)
    result = set()
    for origin in STATES:
        q = deque([(origin, 0)])
        seen_dist = {origin: 0}
        while q:
            u, dist = q.popleft()
            if max_steps is not None and dist >= max_steps:
                continue
            for v in adj[u]:
                if v != origin:
                    result.add((origin, v))
                if v not in seen_dist:
                    seen_dist[v] = dist + 1
                    q.append((v, dist + 1))
    return result


def constructive_candidates(intact_edges, failed_edge):
    """Possibili nuovi archi fisici supportati da un cammino di due passi."""
    existing = set(intact_edges)
    return sorted({(u, w)
                   for u, v in intact_edges
                   for vv, w in intact_edges
                   if v == vv and u != w
                   and (u, w) not in existing
                   and (u, w) != failed_edge})


def simulate_one_system(edges, max_steps=2):
    before = reach(edges, max_steps)
    trials = []
    for i, failed in enumerate(edges):
        remaining = edges[:i] + edges[i + 1:]
        base = reach(remaining, max_steps)
        unlimited = reach(remaining)
        best_edge = None
        best_reach = base
        for candidate in constructive_candidates(remaining, failed):
            # "Candidate" e' raggiungibile gia' prima: la sua costruzione
            # indipendente non puo' espandere la chiusura illimitata.
            assert candidate in unlimited
            constructed = reach(remaining + [candidate], max_steps)
            assert reach(remaining + [candidate]) == unlimited
            if len(constructed) > len(best_reach):
                best_edge = candidate
                best_reach = constructed
        trials.append({
            'failed': failed,
            'baseline': base,
            'with_reconfiguration': best_reach,
            'constructed': best_edge,
            'unlimited': unlimited,
        })
    n = len(trials)
    return {
        'initial_k': len(before),
        'initial_unlimited': len(reach(edges)),
        'mean_after_failure_k': sum(len(t['baseline']) for t in trials) / n,
        'mean_after_reconfiguration_k': sum(len(t['with_reconfiguration']) for t in trials) / n,
        'mean_after_failure_unlimited': sum(len(t['unlimited']) for t in trials) / n,
        'original_retention_baseline': sum(len(t['baseline'] & before) for t in trials) / (n * len(before)),
        'original_retention_reconfigured': sum(len(t['with_reconfiguration'] & before) for t in trials) / (n * len(before)),
        'new_relative_to_initial_k': sum(len(t['with_reconfiguration'] - before) for t in trials) / n,
        'guaranteed_baseline': len(set.intersection(*(t['baseline'] for t in trials))),
        'guaranteed_reconfigured': len(set.intersection(*(t['with_reconfiguration'] for t in trials))),
        'build_frequency': sum(t['constructed'] is not None for t in trials) / n,
        'trials': trials,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-steps', type=int, default=2,
                        help='Numero massimo di operazioni per missione (default 2)')
    parser.add_argument('--details', action='store_true',
                        help='Stampa risultati per ogni guasto')
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error('--max-steps deve essere >= 1')

    print(f'AP-02 | 6 stati; 6 archi; 30 missioni; scadenza K={args.max_steps}')
    print('Guasto uniforme; nuova operazione fisica post-guasto, costo 1,')
    print('se sintetizzabile da un cammino superstite di due archi.')
    print('NESSUNA conoscenza anticipata della missione; riconfigurazione con ottimo esaustivo.\n')
    print(f"{'Sistema':19} {'Prima':>6} {'Guasto':>8} {'Dopo ric.':>10} "
          f"{'P0':>7} {'P1':>7} {'Gar0':>5} {'Gar1':>5} {'E[costo]':>9}")
    print('-' * 90)
    for name, edges in SYSTEMS.items():
        r = simulate_one_system(edges, args.max_steps)
        old = r['mean_after_failure_k']
        new = r['mean_after_reconfiguration_k']
        print(f'{name:19} {r["initial_k"]:6d} {old:8.1f} {new:10.1f} '
              f'{old/30:7.1%} {new/30:7.1%} '
              f'{r["guaranteed_baseline"]:5d} {r["guaranteed_reconfigured"]:5d} '
              f'{r["build_frequency"]:9.2f}')
        if args.details:
            print(f'   Compiti originari preservati dopo guasto: '
                  f'{r["original_retention_baseline"]:.1%}; dopo ric.: '
                  f'{r["original_retention_reconfigured"]:.1%}. '
                  f'Nuovi rispetto al vecchio catalogo con deadline: '
                  f'{r["new_relative_to_initial_k"]:.1f} (media).')
            print(f'   Repertorio senza deadline dopo il guasto: '
                  f'{r["mean_after_failure_unlimited"]:.1f} in media; '
                  f'la ricostruzione con shortcut NON lo modifica.')
            for trial in r['trials']:
                f = ''.join(trial['failed'])
                e = ''.join(trial['constructed']) if trial['constructed'] else '--'
                print(f'   Guasto {f}: {len(trial["baseline"]):2d} -> '
                      f'{len(trial["with_reconfiguration"]):2d} compiti; nuovo arco {e}')
    print('\nP0/P1: frequenza di successo su guasto e missione equiprobabili.')
    print('Gar0/Gar1: numero di missioni garantite sotto ogni guasto singolo.')
    print('E[costo]: quota di guasti che comporta la costruzione di un arco (costo 1).')
    print('Questo e\' un calcolo esatto sul toy model, non una legge fisica generale.')


if __name__ == '__main__':
    main()
