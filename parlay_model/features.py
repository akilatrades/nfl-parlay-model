"""Pre-game features built only from games played before the one being predicted.

Two kinds of player history:

- **Usage** (targets, carries, pass attempts and shares of the team's total):
  exponentially weighted averages with a half-life of HALF_LIFE games, so
  recent games count most.
- **Efficiency** (yards per target, per carry, per attempt; catch rate;
  touchdown rates): totals over the last EFFICIENCY_WINDOW games, shrunk
  toward the position average from earlier seasons. A player with few
  targets is pulled strongly toward the average; a player with many is not.
"""

import numpy as np
import pandas as pd

HALF_LIFE = 4
EFFICIENCY_WINDOW = 24
MIN_PRIOR_GAMES = 3

# (feature, numerator column, denominator column, shrinkage strength)
EFFICIENCY = [
    ("ypt", "receiving_yards", "targets", 30),
    ("catch_rate", "receptions", "targets", 30),
    ("ypc", "rushing_yards", "carries", 40),
    ("ypa", "passing_yards", "attempts", 150),
    ("pass_td_rate", "passing_tds", "attempts", 150),
    ("td_per_touch", "skill_tds", "touches", 60),
]
USAGE = [
    "targets",
    "carries",
    "attempts",
    "target_share",
    "carry_share",
    "att_share",
]
# The player's own recent average of each outcome: the naive baseline.
OUTCOMES = [
    "receiving_yards",
    "receptions",
    "rushing_yards",
    "passing_yards",
    "passing_tds",
    "skill_tds",
]
POSITION_GROUP = {"WR": "WR", "TE": "TE", "RB": "RB", "FB": "RB", "QB": "QB"}


def prior_ewm(frame, column, key="player_id"):
    """EWMA of a column over each player's earlier rows (rows in time order)."""
    return frame.groupby(key)[column].transform(
        lambda s: s.shift().ewm(halflife=HALF_LIFE, min_periods=1).mean()
    )


def prior_sum(frame, column, window=EFFICIENCY_WINDOW, key="player_id"):
    return frame.groupby(key)[column].transform(
        lambda s: s.shift().rolling(window, min_periods=1).sum()
    )


def position_means(frame):
    """League efficiency by position group from seasons before each season.

    The first season falls back to its own average so early training rows
    have a shrinkage target; evaluation seasons only ever see earlier data.
    """
    played = frame[frame.targets.notna()]
    columns = list(dict.fromkeys(c for _, n, d, _ in EFFICIENCY for c in (n, d)))
    totals = played.groupby(["pos_group", "season"])[columns].sum()
    rows = []
    for group, by_season in totals.groupby(level="pos_group"):
        by_season = by_season.droplevel(0).sort_index()
        earlier = by_season.cumsum().shift()
        earlier.iloc[0] = by_season.iloc[0]
        for season, row in earlier.iterrows():
            means = {"pos_group": group, "season": season}
            for name, num, den, _ in EFFICIENCY:
                means[f"mu_{name}"] = row[num] / row[den] if row[den] else np.nan
            rows.append(means)
    return pd.DataFrame(rows)


TEAM_VOLUME = ["team_targets", "team_carries", "team_attempts"]
DEFENSE_HALF_LIFE = 6


def team_features(frame):
    """Each team's recent volume, and what its opponent's defense has allowed.

    Both are EWMAs over earlier games only. Defense uses a longer half-life
    because one game says little about a defense.
    """
    teams = (
        frame[
            ["game_id", "season", "week", "team", "opp"]
            + TEAM_VOLUME
            + ["team_passing_yards", "team_rushing_yards"]
        ]
        .drop_duplicates(["game_id", "team"])
        .sort_values(["season", "week"])
        .reset_index(drop=True)
    )
    for column in TEAM_VOLUME:
        teams[f"ew_{column}"] = prior_ewm(teams, column, key="team")
    # Yards a defense allowed = yards its opponent gained against it.
    for column, name in [
        ("team_passing_yards", "def_pass_allowed"),
        ("team_rushing_yards", "def_rush_allowed"),
    ]:
        teams[name] = teams.groupby("opp")[column].transform(
            lambda s: s.shift().ewm(halflife=DEFENSE_HALF_LIFE, min_periods=1).mean()
        )
    keep = ["game_id", "team"] + [f"ew_{c}" for c in TEAM_VOLUME]
    return teams[keep + ["def_pass_allowed", "def_rush_allowed"]]


def build_features(frame):
    """Add prior-only usage, efficiency and team features to player-game rows.

    ``frame`` holds played games (with outcomes) and, optionally, upcoming
    games (outcomes missing). Upcoming rows get features from played games
    only.
    """
    frame = frame.sort_values(["season", "week", "player_id"]).reset_index(drop=True)
    frame["pos_group"] = frame.position.map(POSITION_GROUP)
    frame["touches"] = frame.targets + frame.carries
    frame["n_prior"] = frame.groupby("player_id").cumcount()

    for column in USAGE + OUTCOMES:
        frame[f"ew_{column}"] = prior_ewm(frame, column)

    means = position_means(frame)
    frame = frame.merge(means, on=["pos_group", "season"], how="left")
    # An upcoming week in a season with no completed games falls back to the
    # latest season with means.
    for name, *_ in EFFICIENCY:
        latest = means.sort_values("season").groupby("pos_group")[f"mu_{name}"].last()
        frame[f"mu_{name}"] = frame[f"mu_{name}"].fillna(frame.pos_group.map(latest))

    for name, num, den, strength in EFFICIENCY:
        numer = prior_sum(frame, num).fillna(0)
        denom = prior_sum(frame, den).fillna(0)
        frame[name] = (numer + strength * frame[f"mu_{name}"]) / (denom + strength)

    frame = frame.merge(team_features(frame), on=["game_id", "team"], how="left")
    return frame


def model_rows(frame, min_prior=MIN_PRIOR_GAMES):
    """Rows with enough history and a market view of the game."""
    keep = (
        (frame.n_prior >= min_prior)
        & frame.implied_total.notna()
        & frame.p_win.notna()
        & frame.ew_team_targets.notna()
        & frame.def_pass_allowed.notna()
    )
    return frame[keep].copy()
