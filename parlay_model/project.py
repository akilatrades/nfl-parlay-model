"""Single-leg projections for an upcoming week."""

import numpy as np
import pandas as pd

from parlay_model.data import player_games, upcoming_rows
from parlay_model.features import build_features, model_rows
from parlay_model.marginals import STATS, fit_all

PERCENTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
YARDAGE = ["receiving_yards", "receptions", "rushing_yards", "passing_yards"]


def project_week(games, players, season, week):
    """Fit on every game before ``week`` and project each eligible player.

    Returns one row per player and stat with the expected value, the
    10th-90th percentiles and, for touchdowns, the chance of each count.
    """
    played = player_games(games, players)
    played = played[(played.season < season) | (played.week < week)]
    upcoming = upcoming_rows(games, players, season, week)
    frame = build_features(pd.concat([played, upcoming], ignore_index=True))
    rows = model_rows(frame)
    train = rows[rows.played.astype(bool)]
    target = rows[(rows.season == season) & (rows.week == week)]
    models = fit_all(train)

    out = []
    for name, model in models.items():
        spec = STATS[name]
        eligible = target[spec.eligible(target)]
        if eligible.empty:
            continue
        mean = model.mean(eligible)
        record = pd.DataFrame(
            {
                "game_id": eligible.game_id.to_numpy(),
                "team": eligible.team.to_numpy(),
                "opp": eligible.opp.to_numpy(),
                "player": eligible.player_display_name.to_numpy(),
                "player_id": eligible.player_id.to_numpy(),
                "position": eligible.position.to_numpy(),
                "stat": name,
                "expected": np.round(mean, 2),
            }
        )
        if name in YARDAGE:
            qs = model.quantiles(eligible, PERCENTILES, mean)
            for i, q in enumerate(PERCENTILES):
                record[f"p{int(q * 100)}"] = np.round(qs[:, i], 1)
        else:
            for k in [1, 2, 3]:
                record[f"prob_{k}plus"] = np.round(
                    model.prob_over(eligible, k - 0.5, mean), 4
                )
        out.append(record)
    return pd.concat(out, ignore_index=True), models
