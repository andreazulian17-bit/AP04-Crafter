#!/usr/bin/env python3
"""AP-04H3C — controllo positivo per l'acquisizione di collect_wood in Crafter.

NON testa H3, Gamma o antifragilita'. In ogni episodio viene preparata una
micro-situazione controllata: una 'tree' oppure una 'stone' davanti al giocatore.
Il learner riceve un *sensore semantico privilegiato* della casella frontale:
NON e' un agente visivo, ne' un agente che scopre nuove possibilita'. Impara
invece, dal feedback reale di Crafter, quando usare 'do' anziche 'noop'.

Il controllo negativo usa feedback invertito; il controllo frozen non aggiorna.
Training ed evaluation usano seed disgiunti, senza contaminazione dei Q.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

NOOP = 0
MOVE_RIGHT = 2
DO = 5
TRAIN_SEED_BASE = 10000
TEST_SEED_BASE = 20000
TARGET = "collect_wood"


@dataclass
class Trial:
    seed: int
    context: str
    requested: int
    learned_reward: float
    wood_collected: int
    correct_context_action: int
    player_unchanged_during_orientation: int


class ContextBandit:
    """Two-arm contextual bandit. Q is learned, not a prewritten tree->do rule."""

    def __init__(self, alpha: float = 0.35, epsilon: float = 0.45, mode: str = "normal"):
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.mode = mode
        self.q = {"tree": [0.0, 0.0], "stone": [0.0, 0.0]}
        self.updates = 0

    def act(self, context: str, rng: random.Random, learning: bool) -> int:
        if learning and rng.random() < self.epsilon:
            return [NOOP, DO][rng.randrange(2)]
        scores = self.q[context]
        # Tie-break prefissato: non eseguire azioni se il feedback non ne dimostra l'utilita'.
        return DO if scores[1] > scores[0] else NOOP

    def update(self, context: str, requested: int, honest_reward: float):
        if self.mode == "frozen":
            return
        reward = 1.0 - honest_reward if self.mode == "inverted" else honest_reward
        idx = 1 if requested == DO else 0
        old = self.q[context][idx]
        self.q[context][idx] = old + self.alpha * (reward - old)
        self.updates += 1


def unpack_step(data):
    if len(data) == 4:
        obs, reward, done, info = data
    elif len(data) == 5:
        obs, reward, terminated, truncated, info = data
        done = terminated or truncated
    else:
        raise ValueError(f"API di step inattesa: {len(data)} valori")
    return obs, float(reward), bool(done), info or {}


def trial(factory, seed: int, context: str, agent: ContextBandit,
          training: bool = False) -> Trial:
    if context not in ("tree", "stone"):
        raise ValueError(context)
    env = factory(seed)
    try:
        if list(env.action_names)[:6] != [
            "noop", "move_left", "move_right", "move_up", "move_down", "do"
        ]:
            raise AssertionError("Ordine delle azioni incompatibile con Crafter originale")
        env.reset()
        player = env._player
        world = env._world
        pos = tuple(map(int, player.pos))
        target = (pos[0] + 1, pos[1])
        if not (0 <= target[0] < int(world.area[0]) and
                0 <= target[1] < int(world.area[1])):
            raise RuntimeError("Spawn troppo vicino al confine")

        # Costruzione *esplicita* di micro-scena controllata. Questa e' una
        # modifica didattica al mondo Crafter, non una perturbazione di H3.
        world[target] = context
        material, _ = world[target]
        assert material == context, "Il motore non ha accettato il setup"

        # L'azione di movimento orienta l'agente a destra, ma tree/stone
        # sono materiali non attraversabili: la posizione deve rimanere fissa.
        _, _, done, orient_info = unpack_step(env.step(MOVE_RIGHT))
        still = tuple(map(int, player.pos)) == pos
        if not still or done:
            raise AssertionError("Orientamento non controllato: posizione/terminale imprevisti")
        if tuple(map(int, player.facing)) != (1, 0):
            raise AssertionError("L'agente non guarda verso destra")

        # Il SENSORE usa dati privilegiati dell'ambiente solo in questo test
        # tecnico. Non e' un input visivo RGB; non confondere con un H3 vero.
        observed_context = world[target][0]
        assert observed_context == context
        rng = random.Random(seed * 1009 + (7 if training else 17))
        requested = agent.act(observed_context, rng, training)
        _, _, _, info = unpack_step(env.step(requested))
        collected = int((info.get("achievements") or {}).get(TARGET, 0) > 0)
        invwood = int((info.get("inventory") or {}).get("wood", 0) > 0)
        if collected != invwood:
            raise AssertionError("Disaccordo collect_wood/inventario: setup non valido")
        if collected and not (context == "tree" and requested == DO):
            raise AssertionError("collect_wood apparso senza interazione con albero")
        # Feedback del compito: reward non e' quello globale Crafter;
        # premia la decisione contestualmente appropriata.
        correct = int((context == "tree" and requested == DO and collected == 1) or
                      (context == "stone" and requested == NOOP))
        if training:
            agent.update(observed_context, requested, float(correct))
        return Trial(seed, context, requested, float(correct), collected,
                     correct, int(still))
    finally:
        if hasattr(env, "close"):
            env.close()


def evaluate(factory, agent, test_worlds, label):
    rows = []
    for i, seed in enumerate(test_worlds):
        # Stessi mondi per entrambe le condizioni (paired test), con seme
        # indipendente dall'addestramento.
        for context in ("tree", "stone"):
            r = trial(factory, seed, context, agent, training=False)
            rows.append({"stage": label, "seed": seed, "context": context,
                         "action": r.requested, "wood_collected": r.wood_collected,
                         "context_correct": r.correct_context_action})
    return rows


def metrics(rows):
    tree = [r for r in rows if r["context"] == "tree"]
    stone = [r for r in rows if r["context"] == "stone"]
    if not tree or not stone:
        raise AssertionError("Prove positive/negative assenti")
    return {
        "n_heldout_worlds": len(tree),
        "wood_success_tree": round(sum(x["wood_collected"] for x in tree) / len(tree), 6),
        "no_unnecessary_interaction_stone": round(
            sum(x["context_correct"] for x in stone) / len(stone), 6),
        "context_correct_overall": round(
            sum(x["context_correct"] for x in rows) / len(rows), 6),
    }


def run(factory, output, train_count=48, test_count=16):
    if train_count < 16 or train_count % 2:
        raise ValueError("train_count deve essere pari e >= 16")
    if test_count < 4:
        raise ValueError("test_count deve essere >= 4")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    train_seeds = range(TRAIN_SEED_BASE, TRAIN_SEED_BASE + train_count)
    test_seeds = range(TEST_SEED_BASE, TEST_SEED_BASE + test_count)
    records, summaries = [], []
    for agent_name, mode in (("frozen", "frozen"),
                             ("learner", "normal"),
                             ("inverted_feedback", "inverted")):
        agent = ContextBandit(mode=mode)
        before = evaluate(factory, agent, test_seeds, "before")
        # Un'opportunita' per episodio; training su contesti alternati.
        for i, seed in enumerate(train_seeds):
            context = "tree" if i % 2 == 0 else "stone"
            trial(factory, seed, context, agent, training=True)
        after = evaluate(factory, agent, test_seeds, "after")
        for row in before + after:
            records.append({"agent": agent_name, **row})
        m0 = metrics(before)
        m1 = metrics(after)
        gained = int(m0["wood_success_tree"] < 0.8 and
                     m1["wood_success_tree"] >= 0.8 and
                     m1["no_unnecessary_interaction_stone"] >= 0.8)
        summary = {
            "agent": agent_name,
            "feedback_mode": mode,
            "train_attempts": train_count,
            "learning_updates": agent.updates,
            "before_wood_success": m0["wood_success_tree"],
            "after_wood_success": m1["wood_success_tree"],
            "before_stone_specificity": m0["no_unnecessary_interaction_stone"],
            "after_stone_specificity": m1["no_unnecessary_interaction_stone"],
            "new_controlled_wood_capacity": gained,
            "gamma_positive_control_per_training_attempt": round(gained/train_count, 8),
            "q_tree_noop": round(agent.q["tree"][0], 6),
            "q_tree_do": round(agent.q["tree"][1], 6),
            "q_stone_noop": round(agent.q["stone"][0], 6),
            "q_stone_do": round(agent.q["stone"][1], 6),
        }
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False))

    with (out / "trials.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    with (out / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)
    report = {
        "experiment": "AP04H3C_POSITIVE_CONTROL_NOT_H3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "engine": "crafter_1.8.3_patched" if factory.__name__ == "real_factory" else "mock_only",
        "train_seed_range": [TRAIN_SEED_BASE, TRAIN_SEED_BASE + train_count - 1],
        "heldout_seed_range": [TEST_SEED_BASE, TEST_SEED_BASE + test_count - 1],
        "context_setup": "Intervento sperimentatore: fronte tree oppure stone, posizione e orientamento controllati",
        "student_sensor": "Privileged one-cell semantic read of world, NOT RGB; no hidden success label",
        "task": TARGET,
        "decision_actions": {"noop": NOOP, "do": DO},
        "capability_thresholds_pre_registered": {"wood_tree_success_min": 0.8,
                                                  "stone_specificity_min": 0.8,
                                                  "baseline_before_wood_below": 0.8},
        "important_limitation": "No autonomous navigation, no unknown tasks, no survival study; positive control only",
        "agents": summaries,
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


# Minimal mock: exercises the API and the entire training/eval pipeline.
class _MockWorld:
    area = (64, 64)
    def __init__(self): self.mat = {}
    def __getitem__(self, k): return self.mat.get(tuple(k), "grass"), None
    def __setitem__(self, k, v): self.mat[tuple(k)] = v


class _MockPlayer:
    pos = (32, 32)
    facing = (0, 1)


class MockEnv:
    action_names = ["noop", "move_left", "move_right", "move_up", "move_down",
                    "do", "sleep", "place_stone", "place_table", "place_furnace",
                    "place_plant", "make_wood_pickaxe", "make_stone_pickaxe",
                    "make_iron_pickaxe", "make_wood_sword", "make_stone_sword", "make_iron_sword"]
    def __init__(self, seed): self.seed = seed; self.reset()
    def reset(self):
        self._world = _MockWorld()
        self._player = _MockPlayer()
        self._collected = 0
        return None
    def step(self, action):
        if action == MOVE_RIGHT:
            self._player.facing = (1, 0)
        if action == DO:
            x, y = self._player.pos
            if self._world[(x+1,y)][0] == "tree":
                self._collected = 1
                self._world[(x+1,y)] = "grass"
        info = {"achievements": {TARGET: self._collected},
                "inventory": {"wood": self._collected}}
        return None, float(self._collected), False, info
    def close(self): pass


def mock_factory(seed): return MockEnv(seed)


def real_factory(seed):
    import crafter
    return crafter.Env(seed=seed, length=10)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--output", default="risultati_h3c")
    p.add_argument("--train", type=int, default=48)
    p.add_argument("--test", type=int, default=16)
    args = p.parse_args()
    if args.self_test:
        data = run(mock_factory, args.output, train_count=48, test_count=8)
        a, b, c = data["agents"]
        assert a["after_wood_success"] == 0.0
        assert b["after_wood_success"] == 1.0
        assert b["after_stone_specificity"] == 1.0
        assert c["after_wood_success"] == 0.0
        assert b["new_controlled_wood_capacity"] == 1
        print("SELF-TEST MOCK OK (NON SONO DATI CRAfter REALI)")
    else:
        from importlib.metadata import version
        if version("crafter") != "1.8.3":
            raise RuntimeError("Richiesta la versione crafter 1.8.3")
        run(real_factory, args.output, args.train, args.test)


if __name__ == "__main__":
    main()
