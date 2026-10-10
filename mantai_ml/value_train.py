"""Fit search's value model to engine-rollout returns, with whole-game validation.

Search continues hidden deals with strategy-engine players, so these targets
describe that continuation policy rather than a different sampling policy.
"""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

from .features import STATE_SIZE, encode_state, state_from_game
from .model import load_model
from .search import critic_path, _load_critic, _mlp


def _value_games(seeds):
    import mahjong_simulator as sim
    rows = []
    for seed in seeds:
        session = sim.GameSession(seed=seed, check=True)
        game = session.game
        game.player1 = 'engine'
        observations = []
        while not session.is_over:
            decision = session.decision
            if decision.player == 0 and decision.kind == 'turn' and any(
                    action.kind == 'discard' for action in decision.actions):
                observations.append((encode_state(**state_from_game(game, 0)), game.instant[0]))
            session.apply(sim.choose_action(game, decision))
        total = session.result()['total'][0]
        rows.extend((x, float(total-before)) for x, before in observations)
    return rows


def train_value(model_path, out_path=None, games=2000, val_games=500, epochs=20,
                batch_size=512, lr=3e-4, workers=4, seed=0, log=print):
    if min(games, val_games, epochs, batch_size, workers) < 1:
        raise ValueError('games, val_games, epochs, batch_size and workers must be positive')
    policy, _ = load_model(model_path)
    out_path = Path(out_path or str(Path(model_path).with_suffix('')) + '_critic.pt')
    old_path = critic_path(model_path)
    old_layers = _load_critic(old_path)
    rng = random.Random(seed)
    seeds = rng.sample(range(2**31), games + val_games)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        def collect(group):
            jobs = [group[i:i+100] for i in range(0, len(group), 100)]
            rows = []
            for batch in pool.map(_value_games, jobs):
                rows.extend(batch)
            return rows
        training = collect(seeds[:games])
        validation = collect(seeds[games:])
    if not training or not validation:
        raise ValueError('No discard positions were collected')
    log(f'Value training: {len(training)} positions / {games} games; '
        f'validation: {len(validation)} positions / {val_games} separate games')
    x = torch.tensor(np.stack([r[0] for r in training]), dtype=torch.float32)
    y = torch.tensor([r[1] for r in training], dtype=torch.float32)
    vx = torch.tensor(np.stack([r[0] for r in validation]), dtype=torch.float32)
    vy = torch.tensor([r[1] for r in validation], dtype=torch.float32)
    torch.manual_seed(seed)
    hidden = policy.hidden
    critic = nn.Sequential(nn.Linear(STATE_SIZE, hidden), nn.ReLU(),
                           nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Linear(hidden//2, 1))
    if old_path:
        critic.load_state_dict(torch.load(old_path, map_location='cpu'))
    optimizer = torch.optim.Adam(critic.parameters(), lr=lr)
    if old_layers:
        old_predictions = torch.tensor(_mlp(vx.numpy(), old_layers)[:, 0])
        baseline_mse = float(((old_predictions-vy)**2).mean())
    else:
        baseline_mse = float(((y.mean()-vy)**2).mean())
    best_mse = baseline_mse
    best_state = deepcopy(critic.state_dict()) if old_path else None
    best_epoch = 0
    history = []
    for epoch in range(1, epochs+1):
        critic.train()
        order = torch.randperm(len(x))
        for first in range(0, len(order), batch_size):
            indices = order[first:first+batch_size]
            loss = ((critic(x[indices]).squeeze(-1)-y[indices])**2).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            optimizer.step()
        critic.eval()
        with torch.no_grad():
            mse = float(((critic(vx).squeeze(-1)-vy)**2).mean())
        history.append(dict(epoch=epoch, validation_rmse=mse**0.5))
        if np.isfinite(mse) and mse < best_mse:
            best_state, best_mse, best_epoch = deepcopy(critic.state_dict()), mse, epoch
        log(f'Value epoch {epoch:2d}: validation RMSE {mse**0.5:.3f} units')
    report = dict(seed=seed, training_games=games, validation_games=val_games,
                  training_positions=len(training), validation_positions=len(validation),
                  baseline_rmse=baseline_mse**0.5, selected_rmse=best_mse**0.5,
                  selected_epoch=best_epoch, baseline='previous critic' if old_path else 'training mean',
                  improved=best_epoch > 0, history=history)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        torch.save(best_state, out_path)
    report['saved'] = best_state is not None
    out_path.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    log(f'Value baseline RMSE {report["baseline_rmse"]:.3f}; '
        f'selected {report["selected_rmse"]:.3f}; epoch {best_epoch}')
    return report
