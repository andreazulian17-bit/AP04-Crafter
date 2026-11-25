#!/usr/bin/env python3
"""AP-04H3 — PILOTA di apprendimento su Crafter 1.8.3 (+ patch locale).

Non dimostra H3 e NON stima una grandezza fisica. Il protocollo confronta:
  frozen: stessa politica iniziale ma senza aggiornamento dei pesi;
  slow: aggiornamento online lento;
  fast: aggiornamento online più veloce.

Misura, su mondi indipendenti dall'allenamento:
  - proxy di competenza = achievements ripetuti in >=2 degli 8 episodi-probe
  - delta del proxy fra prima e dopo training
  - persistenza su test separati, con remap e interruzioni comandi.

La politica è un semplice contextual bandit su inventario/osservazione; non una AGI.
Dipendenze: Python standard library + crafter (numpy come dipendenza di crafter).
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

N_TRAIN=32
N_PROBES=8
N_TEST=8
STEPS=80
SHOCK_AT=40
PROBE_THRESHOLD=2


def obs_key(obs, info):
    # Features low-dimensional, held fixed across groups. A partial observation
    # is all the agent gets; no access to hidden state or future perturbations.
    inv = (info or {}).get('inventory', {}) or {}
    bits = []
    for item in ['health','food','drink','energy']:
        v = int(inv.get(item, 0))
        bits.append(0 if v<=3 else (1 if v<=7 else 2))
    for item in ['wood','stone','wood_pickaxe','stone_pickaxe']:
        bits.append(int(inv.get(item,0)>0))
    # Coarse visual input; uses image only, not privileged game state.
    try:
        h, w = obs.shape[:2]
        pixel = obs[h//2, w//2]
        bits.extend(int(float(x)//64) for x in pixel[:3])
    except (AttributeError, IndexError, TypeError):
        bits.extend((0,0,0))
    return tuple(bits)


class Agent:
    def __init__(self, alpha, n_actions=17):
        self.alpha=float(alpha)
        self.n_actions=n_actions
        self.q=defaultdict(lambda: [0.0]*n_actions)
        self.updates=0

    def select(self, state, step, rng, training):
        # Equal initial behavior for all treatments: deterministic broad action cycle.
        # A learned preference overrides that cycle only after a positive experience.
        base=step%self.n_actions
        if training and rng.random()<0.22:
            return rng.randrange(self.n_actions)
        vals=self.q[state]
        top=max(vals)
        if top>0.22:
            return vals.index(top)
        return base

    def learn(self, state, action, signal):
        if self.alpha<=0:
            return
        row=self.q[state]
        row[action] += self.alpha*(signal-row[action])
        self.updates+=1


def unpack_reset(result):
    if isinstance(result,tuple) and len(result)==2 and isinstance(result[1],dict):
        return result[0],result[1]
    return result,{}


def unpack_step(result):
    if len(result)==4:
        obs,reward,done,info=result
    elif len(result)==5:
        obs,reward,terminated,truncated,info=result
        done=bool(terminated or truncated)
    else:
        raise RuntimeError('Crafter API step inattesa')
    return obs,float(reward),bool(done),(info or {})


def shaped_signal(reward, previous_info, info, done, step, horizon):
    # Agent update signal uses observed consequences; evaluation never uses it.
    prev=(previous_info or {}).get('inventory',{}) or {}
    curr=(info or {}).get('inventory',{}) or {}
    gain=0.0
    for k in ['wood','stone','coal','iron','sapling','food','drink']:
        gain+=max(0,int(curr.get(k,0))-int(prev.get(k,0)))
    hp_delta=int(curr.get('health',0))-int(prev.get('health',0))
    value=reward + 0.10*min(gain,4) + 0.05*hp_delta
    if done and step+1<horizon:
        value-=0.3
    return max(-2.0,min(2.0,value))


def actual_action(requested, step, n_actions, condition, shock_at):
    if step<shock_at:
        return requested
    if condition=='remap':
        return (requested+1)%n_actions
    if condition=='pulse':
        # periodic actuator failure; distinct from permanent complete lockout
        return 0 if (step-shock_at)%4==0 else requested
    return requested


def episode(env_factory, agent, seed, condition='baseline', training=False, horizon=STEPS, shock_at=SHOCK_AT):
    env=env_factory(seed,horizon)
    rng=random.Random(seed+9119)  # independent from game RNG
    try:
        obs,info=unpack_reset(env.reset())
        n=int(env.action_space.n)
        if n!=agent.n_actions:
            raise RuntimeError(f'Actions {n} vs agent {agent.n_actions}')
        seen=set()
        seen_at_shock=set()
        steps=0
        postshock_alive_steps=0
        for t in range(horizon):
            state=obs_key(obs,info)
            requested=agent.select(state,t,rng,training)
            performed=actual_action(requested,t,n,condition,shock_at)
            new_obs,reward,done,new_info=unpack_step(env.step(performed))
            if training:
                # Learner knows action requested and observation, not future shocks.
                agent.learn(state,requested,shaped_signal(reward,info,new_info,done,t,horizon))
            a=new_info.get('achievements',{}) or {}
            seen.update(k for k,v in a.items() if v)
            if t==shock_at-1:
                seen_at_shock=set(seen)
            if t>=shock_at:
                postshock_alive_steps+=1
            obs,info=new_obs,new_info
            steps=t+1
            if done:
                break
        inv=info.get('inventory',{}) or {}
        hp=int(inv.get('health',0))
        alive_at_end=(steps>=horizon and hp>0)
        return {'seed':seed,'condition':condition,'steps':steps,
                'health_end':hp,'alive_at_horizon':int(alive_at_end),
                'postshock_steps':postshock_alive_steps,
                'achievements':sorted(seen),
                'achievements_after_shock':len(seen-seen_at_shock),
                'total_achievements':len(seen)}
    finally:
        if hasattr(env,'close'):env.close()


def measure(agent, factory, seeds, condition='baseline'):
    records=[episode(factory,agent,s,condition=condition,training=False) for s in seeds]
    counts=defaultdict(int)
    for r in records:
        for a in r['achievements']:
            counts[a]+=1
    repeatable=sorted(k for k,n in counts.items() if n>=PROBE_THRESHOLD)
    return {'records':records,'repeatable':repeatable,'counts':dict(counts)}


def mean(values):
    return sum(values)/len(values) if values else float('nan')


def run(factory, output, train_count, probe_count, test_count, horizon):
    training_seeds=range(1000,1000+train_count)
    probe_seeds=range(2000,2000+probe_count)
    test_seeds=range(3000,3000+test_count)
    agents={'frozen':Agent(0.0),'slow':Agent(0.12),'fast':Agent(0.5)}
    all_episodes=[]
    rows=[]
    for name,agent in agents.items():
        pre=measure(agent,factory,probe_seeds)
        for seed in training_seeds:
            episode(factory,agent,seed,training=True,horizon=horizon,shock_at=horizon//2)
        post=measure(agent,factory,probe_seeds)
        # proxy is number of repeatably unlocked achievements, not discovered physics
        gamma=(len(post['repeatable'])-len(pre['repeatable']))/(train_count*horizon)
        for cond in ['baseline','remap','pulse']:
            runs=[episode(factory,agent,seed,condition=cond,training=False,
                          horizon=horizon,shock_at=horizon//2) for seed in test_seeds]
            for r in runs:
                all_episodes.append({'agent':name,**{k:v for k,v in r.items() if k!='achievements'}})
            rows.append({
                'agent':name,'condition':cond,'alpha':agent.alpha,
                'train_steps':train_count*horizon,'learning_updates':agent.updates,
                'pre_repeatable':len(pre['repeatable']),
                'post_repeatable':len(post['repeatable']),
                'delta_repeatable':len(post['repeatable'])-len(pre['repeatable']),
                'gamma_proxy_per_step':gamma,
                'mean_postshock_steps':mean([r['postshock_steps'] for r in runs]),
                'survival_fraction':mean([r['alive_at_horizon'] for r in runs]),
                'mean_health_end':mean([r['health_end'] for r in runs]),
                'mean_postshock_new_achievements':mean([r['achievements_after_shock'] for r in runs]),
            })
        print(f'{name}: repeatable {len(pre["repeatable"])} -> {len(post["repeatable"])} | updates {agent.updates}')
    output.mkdir(parents=True,exist_ok=True)
    (output/'metadati.json').write_text(json.dumps({
        'experiment':'AP04H3_PILOT_NOT_A_TEST_OF_PHYSICAL_LAW',
        'timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'engine':'Crafter 1.8.3 + local determinism patch',
        'train_count':train_count,'probe_count':probe_count,'test_count':test_count,
        'horizon':horizon,'shock_at':horizon//2,
        'training_seeds':list(training_seeds),'probe_seeds':list(probe_seeds),
        'test_seeds':list(test_seeds),'capability_rule':f'achievement on at least {PROBE_THRESHOLD} independent probe seeds',
        'note':'gamma is an operational proxy, NOT physical repertoire expansion; actions shocks are interface perturbations, NOT physical laws',
        'causal_caveat':'Agent policy changed by training; tests held-out, but only three predefined learning-rate treatments. Not a causal estimate of Gamma.',
        'analysis_policy':'descriptive only; no significance test nor universal conclusion',
    },indent=2),encoding='utf8')
    with (output/'sintesi_pilota.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    with (output/'episodi_test.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=all_episodes[0].keys());w.writeheader();w.writerows(all_episodes)
    print(json.dumps(rows,indent=2))
    return rows


class MockSpace:n=17
class MockEnv:
    action_space=MockSpace()
    def __init__(self,seed,length):
        self.rng=random.Random(seed);self.t=0;self.length=length
        self.inv={'health':9,'food':9,'drink':9,'energy':9,'wood':0,'stone':0}
    def reset(self):return [[0]],{}
    def step(self,action):
        self.t+=1
        if action==5 and self.rng.random()<0.45:self.inv['wood']+=1
        if action==6 and self.rng.random()<0.25:self.inv['food']=min(9,self.inv['food']+1)
        done=self.t>=self.length
        info={'inventory':dict(self.inv),'achievements':{'collect_wood':int(self.inv['wood']>0)}}
        return [[self.t%5]],float(action==5 and self.inv['wood']>0),done,info
    def close(self):pass


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--output',default='risultati_h3')
    p.add_argument('--train',type=int,default=N_TRAIN)
    p.add_argument('--probes',type=int,default=N_PROBES)
    p.add_argument('--tests',type=int,default=N_TEST)
    p.add_argument('--steps',type=int,default=STEPS)
    args=p.parse_args()
    if min(args.train,args.probes,args.tests)<2 or args.steps<20:
        p.error('train, probes, tests >=2; steps >=20')
    if args.self_test:
        s=(2,2,2)
        learner=Agent(0.5)
        frozen=Agent(0.0)
        assert learner.select(s,0,random.Random(1),False)==0
        learner.learn(s,5,2.0)
        frozen.learn(s,5,2.0)
        assert learner.select(s,0,random.Random(1),False)==5
        assert frozen.select(s,0,random.Random(1),False)==0
        assert actual_action(5,40,17,'remap',40)==6
        assert actual_action(5,40,17,'pulse',40)==0
        run(lambda s,h:MockEnv(s,h),Path(args.output),4,4,3,24)
        print('SELF-TEST MOCK COMPLETATO: NON sono dati Crafter.')
    else:
        import crafter
        from importlib.metadata import version
        assert version('crafter')=='1.8.3','Attesa Crafter 1.8.3; aggiornare solo dopo nuova validazione'
        run(lambda s,h:crafter.Env(seed=s,length=h),Path(args.output),
            args.train,args.probes,args.tests,args.steps)
if __name__=='__main__':main()
