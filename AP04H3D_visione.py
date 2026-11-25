#!/usr/bin/env python3
"""AP-04H3D: ponte visivo su Crafter (NON e' un test di H3).

L'agente usa SOLO osservazioni RGB 64x64 e feedback ottenuto dai tentativi.
Il laboratorio configura quattro corridoi di due caselle: tre finiscono su
pietra, uno su albero. L'agente sceglie una direzione; una macro fissa esegue
move, move (bloccata dal materiale), do. Non riceve il nome dell'oggetto,
le coordinate, il semantico, il materiale o la posizione corretta.

Training = quattro tentativi autonomi a reset per ciascun mondo, con feedback
reale collect_wood dopo l'esecuzione; learner classifica le texture RGB delle
quattro caselle di arrivo. Test = una sola scelta, politica congelata.

Il progetto sperimentale richiede interventi ambientali espliciti del tecnico.
Non e' esplorazione libera, ne' misura Gamma, ne' validazione di antifragilita'.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))  # dx,dy
MOVE = (2, 4, 1, 3)  # right,down,left,up (Crafter)
DO = 5
TRAIN_BASE = 30000
TEST_BASE = 40000
TASK = 'collect_wood'


def unpack(result):
    if len(result) == 4:
        obs, reward, done, info = result
    elif len(result) == 5:
        obs, reward, term, trunc, info = result
        done = term or trunc
    else:
        raise ValueError(f'API step non riconosciuta: {len(result)}')
    return obs, float(reward), bool(done), info or {}


def geometry_check(env):
    # Questa geometria e' la documentazione della camera, NON una lettura
    # semantica del mondo passata all'agente.
    if tuple(env.observation_space.shape) != (64, 64, 3):
        raise ValueError('Attesa camera RGB 64x64x3 Crafter')
    if tuple(env.action_names[:6]) != ('noop', 'move_left', 'move_right',
                                       'move_up', 'move_down', 'do'):
        raise ValueError('Interfaccia comandi inattesa')


def setup(factory, seed: int, true_dir: int, shift: int):
    env = factory(seed)
    env.reset()
    geometry_check(env)
    world = env._world   # SOLO laboratorio, proibito all'agente
    player = env._player
    p = tuple(int(z) for z in player.pos)
    offsets = ((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1))
    ox, oy = offsets[shift % len(offsets)]
    desired = (p[0] + ox, p[1] + oy)
    # Pulizia locale simmetrica; non spostare mai il giocatore su un altro obj.
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            v = (desired[0] + dx, desired[1] + dy)
            if v == p:
                continue
            material, obj = world[v]
            if obj is not None and obj is not player:
                world.remove(obj)
            world[v] = 'grass'
    if desired != p:
        if world[desired][1] is not None:
            raise AssertionError('La casella iniziale e occupata')
        world.move(player, desired)
    for i, (dx, dy) in enumerate(DIRECTIONS):
        target = (desired[0] + 2 * dx, desired[1] + 2 * dy)
        world[target] = 'tree' if i == true_dir else 'stone'
    for i, (dx, dy) in enumerate(DIRECTIONS):
        assert world[(desired[0] + dx, desired[1] + dy)][0] == 'grass'
        assert world[(desired[0] + 2 * dx, desired[1] + 2 * dy)][0] == ('tree' if i == true_dir else 'stone')
    image = env.render()  # l'agente riceve soltanto questo array RGB
    return env, image, desired


def tile_features(rgb):
    """Trasformazione predefinita da pixel, senza riconoscimento semantico."""
    rgb = np.asarray(rgb)
    if rgb.shape != (64, 64, 3):
        raise ValueError(f'RGB inatteso: {rgb.shape}')
    # Crafter default 9x9 griglia, 7px/cella, 9x7 mappa piu' inventario.
    # Renderer: transpose finale dopo concatenazione della vista locale.
    # Avatar al centro delle 9x7 celle: (4,3). Materiale a distanza 2.
    out = []
    for dx, dy in DIRECTIONS:
        cx, cy = 4 + 2 * dx, 3 + 2 * dy
        x, y = 7 * cx, 7 * cy
        patch = rgb[y + 1:y + 6, x + 1:x + 6, :].astype(np.float64) / 255.0
        if patch.shape != (5, 5, 3):
            raise AssertionError('Ritaglio non valido')
        # media e deviazione, oltre ai 3x3 valori medi spaziali;
        # caratteristiche di colore e trama, non etichette del materiale.
        feat = np.concatenate((patch.mean(axis=(0, 1)),
                               patch.std(axis=(0, 1)),
                               np.asarray([patch[:2,:2].mean(axis=(0,1)),
                                           patch[:2,-2:].mean(axis=(0,1)),
                                           patch[-2:,:2].mean(axis=(0,1)),
                                           patch[-2:,-2:].mean(axis=(0,1))]).ravel()))
        out.append(feat)
    return np.stack(out)


class PixelLearner:
    def __init__(self):
        self.pos = []
        self.neg = []
        self.mean_pos = self.mean_neg = None
        self.scale = None

    def observe(self, pixel_vec, wood_success):
        (self.pos if wood_success else self.neg).append(np.array(pixel_vec, copy=True))

    def fit(self):
        if not self.pos or not self.neg:
            raise RuntimeError('Mancano esempi positivi o negativi')
        pos, neg = np.stack(self.pos), np.stack(self.neg)
        self.mean_pos = pos.mean(axis=0)
        self.mean_neg = neg.mean(axis=0)
        # Scala tra-classi: evita dominanza di una sola caratteristica.
        self.scale = np.maximum(0.03, np.std(np.concatenate((pos, neg)), axis=0))

    def choose(self, features):
        if self.mean_pos is None:
            return 0
        dpos = np.square((features - self.mean_pos) / self.scale).mean(axis=1)
        dneg = np.square((features - self.mean_neg) / self.scale).mean(axis=1)
        return int(np.argmax(dneg - dpos))


def execute(env, origin, selected):
    direction = MOVE[selected]
    dx, dy = DIRECTIONS[selected]
    _, _, done, _ = unpack(env.step(direction))
    expected = (origin[0] + dx, origin[1] + dy)
    if tuple(map(int, env._player.pos)) != expected or done:
        raise AssertionError('Corridoio non attraversabile: setup non valido')
    _, _, done, _ = unpack(env.step(direction))
    if tuple(map(int, env._player.pos)) != expected or done:
        raise AssertionError('Ostacolo finale non ha fermato l agente')
    _, _, _, info = unpack(env.step(DO))
    ach = int((info.get('achievements') or {}).get(TASK, 0) > 0)
    inv = int((info.get('inventory') or {}).get('wood', 0) > 0)
    if ach != inv:
        raise AssertionError('Disaccordo fra achievement e inventario')
    return ach


def evaluate(factory, learner, count, stage, blinded=False):
    rows = []
    for idx in range(count):
        seed = TEST_BASE + idx
        true_dir = (idx * 3) % 4
        shift = idx % 5
        env, rgb, origin = setup(factory, seed, true_dir, shift)
        try:
            vec = tile_features(rgb)
            if blinded:
                # Controllo negativo: relazioni fra immagine e direzione
                # permutate, ma compiti fisici rimangono invariati.
                perm = np.random.default_rng(seed + 5003).permutation(4)
                choice = learner.choose(vec[perm])
                # Non rimappare: rompe deliberatamente il legame pixel->azione fisica.
            else:
                choice = learner.choose(vec)
            success = execute(env, origin, choice)
            if success != int(choice == true_dir):
                raise AssertionError('Il risultato non coincide con la scena controllata')
            rows.append({'stage': stage, 'seed': seed,
                         'true_dir_audit_only': true_dir, 'start_shift_audit_only': shift,
                         'chosen': choice, 'collected_wood': success})
        finally:
            (env.close() if hasattr(env, "close") else None)
    return rows


def run(factory, output, train=48, test=24):
    if train < 20 or train % 4 or test < 16 or test % 4:
        raise ValueError('train multiplo di 4 e >=20; test multiplo di 4 e >=16')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    learner = PixelLearner()
    before = evaluate(factory, learner, test, 'before_frozen')
    n_attempts = 0
    n_wood = 0
    for idx in range(train):
        seed = TRAIN_BASE + idx
        true_dir = (idx * 3) % 4
        shift = idx % 5
        # Esplorazione realizzata su quattro reset indipendenti dello stesso
        # mondo; nessuna etichetta dell'esperimento viene mostrata al modello.
        for candidate in range(4):
            env, rgb, origin = setup(factory, seed, true_dir, shift)
            try:
                feature = tile_features(rgb)[candidate]
                wood = execute(env, origin, candidate)
                if wood != int(candidate == true_dir):
                    raise AssertionError('Ricompensa empirica non coerente')
                learner.observe(feature, wood)
                n_wood += wood
                n_attempts += 1
            finally:
                (env.close() if hasattr(env, "close") else None)
    learner.fit()
    after = evaluate(factory, learner, test, 'after_learner')
    blinded = evaluate(factory, learner, test, 'after_scrambled_pixels', blinded=True)
    frozen = evaluate(factory, PixelLearner(), test, 'after_frozen')
    rows = before + after + blinded + frozen
    with (out / 'trials.csv').open('w', newline='', encoding='utf8') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    rates = {stage: sum(int(r['collected_wood']) for r in rows if r['stage']==stage)/test
             for stage in ('before_frozen','after_learner','after_scrambled_pixels','after_frozen')}
    # Gate fissato a priori; rifiuto della sufficienza di questo learner
    # se non porta vantaggio abbastanza grande o se il controllo negativo
    # suggerisce una possibile scorciatoia invece della visione.
    gate = bool(rates['after_learner'] >= .75 and
                rates['after_learner'] - rates['after_frozen'] >= .35 and
                rates['after_scrambled_pixels'] <= .5)
    result = {
        'experiment': 'AP04H3D_VISUOMOTOR_BRIDGE_NOT_H3',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'engine': 'crafter_1.8.3_patched' if factory.__name__=='real_factory' else 'mock_only',
        'model': 'RGB patch features + centroid classification learned from environment reward',
        'scene': 'four symmetric corridors length 2, one tree and three stones; start offsets vary',
        'agent_input': 'RGB 64x64 only; no semantic, world, player position or correct direction',
        'controller_prior': 'four known cardinal choices; action macro move-move-do; synchronized resets',
        'train_worlds': train, 'training_attempts': n_attempts,
        'training_positive_rewards': n_wood,
        'heldout_worlds_per_arm': test,
        'rates': rates,
        'preregistered_gate': {'learned_at_least': .75, 'advantage_over_frozen_at_least': .35,
                               'scrambled_at_most': .5, 'passed': gate},
        'interpretation': ('Bridge only; no unbounded discovery, no Gamma, no perturbation, no H3.'),
        'decision': ('GO to H3 design using established competent agents' if gate else
                     'STOP handcrafted micro-agent; pivot to published pretrained Crafter agents'),
    }
    (out / 'report.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf8')
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


# A deterministic, toy renderer used ONLY to test the pipeline (not scientific).
class _MockWorld:
    area = (64,64)
    def __init__(self): self.mat = {}
    def __getitem__(self, p): return self.mat.get(tuple(p), 'grass'), None
    def __setitem__(self, p, m): self.mat[tuple(p)] = m
    def move(self, player, p): player.pos = np.array(p)

class _MockPlayer:
    def __init__(self):
        self.pos = np.array((32,32)); self.facing=(0,1)

class MockEnv:
    action_names = ['noop','move_left','move_right','move_up','move_down','do']
    observation_space = type('Space', (), {'shape': (64,64,3)})()
    def __init__(self, seed): self.seed=seed; self.reset()
    def reset(self):
        self._world = _MockWorld(); self._player=_MockPlayer(); self._wood=0
    def render(self):
        arr=np.zeros((64,64,3),dtype=np.uint8)
        # gradient baseline to ensure handling of non-trivial pixel texture
        palettes={'grass': (62,110,60), 'tree':(42,160,50), 'stone': (140,140,135)}
        for ix in range(9):
            for iy in range(7):
                pos=(int(self._player.pos[0])+ix-4,int(self._player.pos[1])+iy-3)
                label=self._world[pos][0]
                c=np.array(palettes.get(label,palettes['grass']),dtype=np.int16)
                rng=np.random.default_rng(self.seed*31+ix*17+iy)
                noise = rng.integers(-4,5,size=(7,7,3),dtype=np.int16)
                arr[iy*7:(iy+1)*7,ix*7:(ix+1)*7]=np.clip(c+noise,0,255).astype(np.uint8)
        return arr
    def step(self, action):
        if action in MOVE:
            k=MOVE.index(action)
            dx,dy=DIRECTIONS[k]
            self._player.facing=(dx,dy)
            target=(int(self._player.pos[0])+dx,int(self._player.pos[1])+dy)
            if self._world[target][0]=='grass':
                self._player.pos=np.array(target)
        if action==DO:
            dx,dy=self._player.facing
            target=(int(self._player.pos[0])+dx,int(self._player.pos[1])+dy)
            if self._world[target][0]=='tree':
                self._world[target]='grass'; self._wood=1
        return None,float(self._wood),False,{'achievements':{TASK:self._wood},
                                            'inventory':{'wood':self._wood}}
    def close(self): pass

def mock_factory(seed): return MockEnv(seed)

def real_factory(seed):
    import crafter
    return crafter.Env(seed=seed,length=12)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--output',default='risultati_h3d')
    p.add_argument('--train',type=int,default=48)
    p.add_argument('--test',type=int,default=24)
    args=p.parse_args()
    if args.self_test:
        r=run(mock_factory,args.output,train=48,test=24)
        assert r['preregistered_gate']['passed'], 'Autotest mock non ha superato gate'
        print('MOCK SELF-TEST OK — NON CRAfter reale')
    else:
        from importlib.metadata import version
        if version('crafter')!='1.8.3':
            raise RuntimeError('Crafter richiesto 1.8.3')
        run(real_factory,args.output,train=args.train,test=args.test)

if __name__=='__main__': main()
