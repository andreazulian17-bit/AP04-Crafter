#!/usr/bin/env python3
"""Aggregate ACTUAL Crafter smoke-test JSON files; refuse mocked data."""
import argparse
import csv
import json
import pathlib
import statistics
import sys

def main():
    p=argparse.ArgumentParser()
    p.add_argument('directory', nargs='?', default='risultati')
    p.add_argument('--allow-mock', action='store_true', help='Only for validating the reporting script')
    a=p.parse_args()
    root=pathlib.Path(a.directory)
    paths=sorted(root.glob('crafter_seed_*.json'))
    if not paths:
        print('ERRORE: nessun file crafter_seed_*.json trovato.', file=sys.stderr)
        return 2
    rows=[]
    for path in paths:
        data=json.loads(path.read_text(encoding='utf-8'))
        engine=str(data.get('tested_engine', ''))
        prov=data.get('engine_provenance', {})
        if not a.allow_mock and (prov.get('kind') != 'installed-crafter' or not prov.get('module_file') or not prov.get('version')):
            raise ValueError(f'{path}: manca la provenienza del modulo Crafter installato')
        if a.allow_mock and prov.get('kind') not in ('installed-crafter','mock'):
            raise ValueError(f'{path}: provenienza non riconosciuta')
        if data.get('test_kind') != 'ENGINE_SUITABILITY_SMOKE_TEST_NOT_H3':
            raise ValueError(f'{path}: tipo di prova inatteso')
        cond={r['condition']:r for r in data['conditions']}
        for key in ('baseline','remap','lockout'):
            if key not in cond: raise ValueError(f'{path}: condizione mancante: {key}')
        b=cond['baseline']; m=cond['remap']; l=cond['lockout']
        interventions=data['action_interface_interventions']
        rows.append({
           'file':path.name,'seed':data['seed'],'engine':engine,
           'riproducibile':bool(data['engine_reproducibility_same_seed_same_actions']),
           'baseline_steps':b['steps_executed'],'remap_steps':m['steps_executed'],'lockout_steps':l['steps_executed'],
           'baseline_reward':round(b['total_reward'],4),
           'remap_reward':round(m['total_reward'],4),
           'lockout_reward':round(l['total_reward'],4),
           'baseline_achievements':b['achievements_unlocked'],
           'remap_achievements':m['achievements_unlocked'],
           'lockout_achievements':l['achievements_unlocked'],
           'remap_changed_action':bool(interventions['remap_actually_changed_action']),
           'lockout_changed_action':bool(interventions['lockout_actually_changed_action']),
           'remap_first_difference':interventions['remap_first_difference_from_baseline_step'],
           'lockout_first_difference':interventions['lockout_first_difference_from_baseline_step'],
        })
    csv_path=root/'sintesi_crafter.csv'
    with csv_path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
    deterministic=sum(r['riproducibile'] for r in rows)
    interventions=sum(r['remap_changed_action'] and r['lockout_changed_action'] for r in rows)
    baseline_avg=statistics.mean(r['baseline_steps'] for r in rows)
    report={
       'TIPO':'SOLO TEST TECNICO DEL MOTORE CRAfter — NESSUN APPRENDIMENTO E NESSUNA PROVA DI H3',
       'test_real_engine':len(rows),'riproducibili':deterministic,
       'interventi_introdotti':interventions,
       'passi_baseline_media':round(baseline_avg,2),
       'criterio_riproducibilita':deterministic==len(rows),
       'criterio_intervento':interventions==len(rows),
       'GO_tecnico_di_base':deterministic==len(rows) and interventions==len(rows),
       'attenzione':'Un GO tecnico non prova che Crafter misuri nuove capacità o persistenza causale.',
    }
    (root/'sintesi_crafter.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(report,indent=2,ensure_ascii=False))
    print(f'CSV: {csv_path.resolve()}')
    return 0 if report['GO_tecnico_di_base'] else 1

if __name__=='__main__':
    raise SystemExit(main())
