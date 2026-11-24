#!/usr/bin/env python3
"""AP-03 - Possibile adiacente sotto perturbazione (toy model).

Esecuzione:
    python AP-03_possibile_adiacente.py
    python AP-03_possibile_adiacente.py --details
    python AP-03_possibile_adiacente.py --ensemble 750
    python AP-03_possibile_adiacente.py --self-test

Requisiti: Python >= 3.9; sola libreria standard.

Sei stati A..F. Un'operazione e' un arco diretto tra due stati.
Un compito (u, v) e' fattibile se esiste una sequenza di archi u -> v.
In AP-01 gli archi venivano solo eliminati; AP-02 accorciava percorsi esistenti.
In AP-03 un modulo di fabbricazione latente (assunto, non derivato!) puo'
aggiungere UNA operazione (u, v) in precedenza irraggiungibile dopo il guasto.
Non puo' ricostruire l'arco esattamente distrutto. Tutti i nuovi archi hanno
costo di fabbricazione 1 unita' di materiale e 1 tick di tempo.
Le risorse per l'unico intervento sono le stesse per tutte le architetture.

Politiche, adottate DOPO aver osservato il guasto ma PRIMA della missione:
- niente: non costruisce nulla.
- conserva: massimizza il numero di compiti del repertorio originale ancora
  eseguibili; non costruisce se non ottiene un guadagno su questa misura.
- esplora: massimizza i compiti non presenti nel repertorio originale; non
  costruisce se non ottiene guadagno di novita' rispetto al pre-guasto.
Parita': secondariamente massimizza la dimensione del repertorio, quindi sceglie
l'arco lessicograficamente minore.

'Persistenza' e' solo un proxy operativo: successo della missione scelta
uniformemente tra le 30 coppie, dopo un guasto uniforme tra i sei archi.
Limite totale della prima missione dopo guasto: 4 tick. L'eventuale costruzione
consuma 1 tick, e quindi restano al massimo 3 transizioni di esecuzione.
Le misure di repertorio illimitato sono mostrate separatamente.

L'agente ha un catalogo predefinito di tutti gli stati e un oracolo che valuta
l'intera chiusura dei percorsi. Quindi non e' creativita' autonoma, non e'
scoperta di compiti non rappresentabili e non e' una simulazione fisica realistica.
"""
from __future__ import annotations
import argparse
import csv
import math
import random
from collections import deque
from itertools import combinations, product
from pathlib import Path

STATES = tuple('ABCDEF')
TASKS = tuple((a, b) for a, b in product(STATES, repeat=2) if a != b)
SYSTEMS = {
    'S1 Specialista': [('A', 'B'), ('B', 'A'), ('A', 'C'), ('C', 'A'), ('B', 'C'), ('C', 'B')],
    'S2 Compositore': [('A', 'B'), ('B', 'D'), ('A', 'C'), ('C', 'D'), ('D', 'E'), ('E', 'F')],
    'S3 Esploratore': [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E'), ('E', 'F'), ('F', 'A')],
}
POLICIES = ('niente', 'conserva', 'esplora')


def reach(edges, max_steps=None):
    """Insieme di trasformazioni ordinate realizzabili, tramite BFS."""
    adj = {s: [] for s in STATES}
    for u, v in edges:
        adj[u].append(v)
    result = set()
    for src in STATES:
        dist = {src: 0}
        queue = deque([src])
        while queue:
            u = queue.popleft()
            if max_steps is not None and dist[u] >= max_steps:
                continue
            for v in adj[u]:
                if v != src:
                    result.add((src, v))
                if v not in dist:
                    dist[v] = dist[u] + 1
                    queue.append(v)
    return result


def floyd_reach(edges):
    """Verifica indipendente (Warshall) della raggiungibilita' illimitata."""
    m = {p: False for p in TASKS}
    for u, v in edges:
        if u != v:
            m[(u, v)] = True
    for k in STATES:
        for i in STATES:
            for j in STATES:
                if i != j and i != k and j != k:
                    m[(i, j)] = m[(i, j)] or (m[(i, k)] and m[(k, j)])
    return {pair for pair in TASKS if m[pair]}


def candidates(surviving, failed):
    """Solo nuove trasformazioni non presenti nella chiusura dopo il guasto."""
    already = reach(surviving)
    return [pair for pair in TASKS if pair not in already and pair != failed]


def choose(surviving, failed, initial, policy):
    """Restituisce arco da fabbricare, chiusura risultante, e uso di risorsa."""
    base = reach(surviving)
    if policy == 'niente':
        return None, base
    score_before = len(base & initial) if policy == 'conserva' else len(base - initial)
    scores = []
    for edge in candidates(surviving, failed):
        new = reach(surviving + [edge])
        # Una operazione prima irraggiungibile e' una vera espansione
        # rispetto al sistema danneggiato: non uno shortcut preesistente.
        assert edge not in base and len(new - base) > 0
        primary = len(new & initial) if policy == 'conserva' else len(new - initial)
        scores.append((primary, len(new), edge, new))
    if not scores:
        return None, base
    highest = max(x[0] for x in scores)
    if highest <= score_before:
        return None, base
    viable = [x for x in scores if x[0] == highest]
    second_best = max(x[1] for x in viable)
    best = min((x for x in viable if x[1] == second_best), key=lambda x: x[2])
    return best[2], best[3]


def trials_for(name, edges, total_ticks=4, fabrication_ticks=1):
    assert len(edges) == 6 and len(set(edges)) == 6
    original = reach(edges)
    rows = []
    for i, failed in enumerate(edges):
        surviving = edges[:i] + edges[i + 1:]
        base = reach(surviving)
        for policy in POLICIES:
            new_edge, after = choose(surviving, failed, original, policy)
            constructing = new_edge is not None
            remaining_ticks = total_ticks - fabrication_ticks if constructing else total_ticks
            if remaining_ticks < 0:
                remaining_ticks = 0
            crisis_reach = reach(surviving + ([new_edge] if constructing else []), remaining_ticks)
            rows.append({
                'system': name, 'failed': ''.join(failed), 'policy': policy,
                'built': ''.join(new_edge) if new_edge else '--',
                'original': len(original), 'damaged': len(base),
                'after': len(after), 'retained_original': len(after & original),
                'genuinely_new_vs_original': len(after - original),
                'recovery_vs_damaged': len((after & original) - base),
                'crisis_success_count': len(crisis_reach),
                'construction_cost': 1 if constructing else 0,
            })
    return rows


def summarize(rows):
    results = []
    for system in SYSTEMS:
        for policy in POLICIES:
            sub = [r for r in rows if r['system'] == system and r['policy'] == policy]
            assert len(sub) == 6
            avg = lambda key: sum(row[key] for row in sub)/len(sub)
            results.append({
                'system': system, 'policy': policy,
                'initial': sub[0]['original'],
                'damaged': avg('damaged'), 'after': avg('after'),
                'retained': avg('retained_original'),
                'new': avg('genuinely_new_vs_original'),
                'recovered': avg('recovery_vs_damaged'),
                'crisis_success': avg('crisis_success_count') / len(TASKS),
                'cost': avg('construction_cost'),
            })
    return results


def pearson(xs, ys):
    mx = sum(xs)/len(xs); my = sum(ys)/len(ys)
    cov = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    den = math.sqrt(sum((x-mx)**2 for x in xs)*sum((y-my)**2 for y in ys))
    return cov/den if den else float('nan')


def ensemble(n=750, seed=20261124):
    """Reti distinte uniformi con 6 archi diretti su 30 possibili."""
    total = math.comb(len(TASKS), 6)
    if n > total:
        raise ValueError(f'Troppe reti: al massimo {total}')
    rnd = random.Random(seed)
    seen = set(); out = []
    while len(out) < n:
        indices = tuple(sorted(rnd.sample(range(len(TASKS)),6)))
        if indices in seen:
            continue
        seen.add(indices)
        edges = [TASKS[k] for k in indices]
        initial = reach(edges)
        damaged = [reach(edges[:i]+edges[i+1:]) for i in range(6)]
        mean_damaged = sum(len(x) for x in damaged)/6
        out.append({
            'gamma': len(initial)/6,
            'retention': mean_damaged/len(initial),
            'success': mean_damaged/len(TASKS),
        })
    return {
        'n': n, 'total_possible_graphs': total, 'seed': seed,
        'correlation_gamma_retention': pearson([v['gamma'] for v in out], [v['retention'] for v in out]),
        'correlation_gamma_success': pearson([v['gamma'] for v in out], [v['success'] for v in out]),
    }


def self_test():
    # Coerenza fra due algoritmi differenti su tutti i sottografi dei tre sistemi.
    for edges in SYSTEMS.values():
        for k in range(7):
            for indices in combinations(range(6), k):
                sub = [edges[i] for i in indices]
                assert reach(sub) == floyd_reach(sub)
        original = reach(edges)
        for failed in edges:
            surviving = edges.copy(); surviving.remove(failed)
            intact = reach(surviving)
            assert intact <= original
            for edge in candidates(surviving, failed):
                assert edge not in intact
                assert reach(surviving + [edge]) > intact
                assert reach(surviving + [edge]) == floyd_reach(surviving + [edge])
    rows = [r for name, edges in SYSTEMS.items() for r in trials_for(name, edges)]
    summary = {(s['system'],s['policy']):s for s in summarize(rows)}
    assert len(rows) == 54
    assert summary[('S1 Specialista','niente')]['retained'] == 6
    assert summary[('S2 Compositore','conserva')]['retained'] == 79/6
    assert summary[('S2 Compositore','esplora')]['new'] == 71/6
    assert summary[('S3 Esploratore','conserva')]['retained'] == 25
    assert summary[('S3 Esploratore','esplora')]['cost'] == 0
    print('TEST OK: sottografi, Warshall, guasti, candidature, 54 prove, risultati campione.')


def main():
    parser = argparse.ArgumentParser(description='AP-03: capacita nuove vs recupero di funzioni')
    parser.add_argument('--details', action='store_true', help='Dettaglio per sistema, guasto e politica')
    parser.add_argument('--ensemble', type=int, default=750, help='Reti per il confronto esplorativo (0 per disabilitare, default 750)')
    parser.add_argument('--seed', type=int, default=20261124, help='Seed pseudocasuale campione')
    parser.add_argument('--self-test', action='store_true', help='Esegue test di coerenza indipendenti')
    parser.add_argument('--csv', type=Path, help='Esporta i 54 risultati per guasto in CSV')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    rows = [r for name, edges in SYSTEMS.items() for r in trials_for(name, edges)]
    print('AP-03 | 6 stati; 6 archi iniziali; 1 guasto; 1 costruzione opzionale')
    print('La costruzione e\' una nuova operazione IRRAGGIUNGIBILE dopo il guasto;')
    print('vietata la ricostruzione esatta del collegamento perso; costo 1 materiale e 1 tick.')
    print('Missione non nota al momento della scelta; deadline totale prima missione = 4 tick.\n')
    print(f"{'Sistema':17} {'Politica':10} {'R0':>3} {'Danno':>7} {'Dopo':>7} {'Conservati':>10} {'Nuovi':>7} {'Recuperati':>10} {'Crisi':>8} {'Costo':>6}")
    print('-'*100)
    for r in summarize(rows):
        print(f"{r['system']:17} {r['policy']:10} {r['initial']:3d} {r['damaged']:7.2f} {r['after']:7.2f} {r['retained']:10.2f} {r['new']:7.2f} {r['recovered']:10.2f} {r['crisis_success']:7.1%} {r['cost']:6.2f}")
    if args.details:
        print('\nDETTAGLIO PER GUASTO')
        for row in rows:
            print(f"{row['system']:17} guasto={row['failed']} strategia={row['policy']:10} "
                  f"arco={row['built']:2} dopo={row['after']:2} "
                  f"ret={row['retained_original']:2} nuovi={row['genuinely_new_vs_original']:2} "
                  f"crisi={row['crisis_success_count']:2}/30")
    if args.csv:
        with args.csv.open('w', newline='', encoding='utf-8') as fp:
            w=csv.DictWriter(fp, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        print('\nCSV salvato:',args.csv)
    if args.ensemble:
        if args.ensemble < 2:
            parser.error('--ensemble deve essere 0 oppure >=2')
        r=ensemble(args.ensemble,args.seed)
        print(f"\nConfronto su {r['n']} reti distinte, estratte uniformemente tra {r['total_possible_graphs']} (seed {r['seed']}).")
        print('r(Pearson) Gamma vs percentuale compiti originari conservati: '
              f"{r['correlation_gamma_retention']:+.4f}")
        print('r(Pearson) Gamma vs successo assoluto su 30 missioni: '
              f"{r['correlation_gamma_success']:+.4f}")
        print('Correlazioni descrittive: non sono relazioni causali o leggi universali.')
    print('\nLIMITI: fabbricatore postulato, catalogo completo noto al modello, costo semplificato,')
    print('nessuna scoperta di stati non rappresentati, nessuna generalizzazione fisica.')


if __name__ == '__main__':
    main()
