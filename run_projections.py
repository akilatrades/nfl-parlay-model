"""Project every player for the next unplayed week of the current season.

Run from the repository root: python run_projections.py [season week]
"""

import sys
from pathlib import Path

import pandas as pd

from parlay_model.data import load
from parlay_model.marginals import FEATURE_NAMES, STATS
from parlay_model.project import project_week

OUT = Path("outputs/projections")


def next_week(games):
    upcoming = games[games.home_score.isna() & games.total_line.notna()]
    first = upcoming.sort_values(["season", "week"]).iloc[0]
    return int(first.season), int(first.week)


def main():
    games, players = load()
    if len(sys.argv) == 3:
        season, week = int(sys.argv[1]), int(sys.argv[2])
    else:
        season, week = next_week(games)
    projections, models = project_week(games, players, season, week)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{season}_week{week:02d}.csv"
    projections.sort_values(["stat", "expected"], ascending=[True, False]).to_csv(
        path, index=False
    )

    coefficients = [
        {"stat": name, "feature": feature, "coefficient": round(value, 4)}
        for name, model in models.items()
        for feature, value in zip(
            FEATURE_NAMES[name], model.coefficients(), strict=True
        )
    ]
    pd.DataFrame(coefficients).to_csv(OUT / "coefficients.csv", index=False)
    counts = projections.groupby("stat").size().reindex(list(STATS))
    print(f"Wrote {path}")
    print(counts.to_string())


if __name__ == "__main__":
    main()
