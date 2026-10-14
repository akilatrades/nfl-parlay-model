# NFL parlay model

[![tests](https://github.com/akilatrades/nfl-parlay-model/actions/workflows/tests.yml/badge.svg)](https://github.com/akilatrades/nfl-parlay-model/actions/workflows/tests.yml)

A same-game parlay pricing model, built and tested in phases. The goal is a fair price for any combination of NFL player and game legs, to compare against prediction-market and sportsbook quotes. It grows out of my [same-game parlay correlation study](https://github.com/akilatrades/nfl-sgp-correlation).

| Phase | What it adds | Status |
|---|---|---|
| **1** | **A full range of outcomes for every player leg: yards, receptions and touchdowns** | **Done (this release)** |
| 2 | Links between legs (correlation), including extreme lines | Next |
| 3 | Pricing engine for any combo, a phone calculator, paper trading | Planned |
| 4 | Historical prop prices to test against real markets | If paper trading justifies it |
| 5 | Drive-level game simulator | Stretch |

## Phase 1 result

**Across 125,803 player-stat predictions from 2019 through week 5 of 2026, each made using only earlier games, the model's probabilities were calibrated to within about one percentage point in every band, and it beat each player's own recent average on all six stats.** The gain is large for quarterbacks and touchdowns and small for receiving and rushing yards, where a player's recent production is already a strong guide.

![Calibration](outputs/backtest/calibration.svg)

| Stat | Player-games | Typical miss: model vs recent average | Brier score improvement | Log loss improvement |
|---|---:|---:|---:|---:|
| Passing yards (starting QB) | 3,713 | **58.7 vs 65.2 yards** | 4.5% | 4.4% |
| Passing TDs (starting QB) | 3,713 | 0.90 vs 0.93 TDs | **5.8%** | **6.0%** |
| Anytime TD (WR, TE, RB) | 35,774 | 0.34 vs 0.34 TDs | 3.7% | 4.1% |
| Receiving yards (WR, TE, RB) | 35,774 | 17.0 vs 17.1 yards | 1.1% | 1.9% |
| Rushing yards (RB) | 11,055 | 19.1 vs 19.1 yards | 0.8% | 1.2% |
| Receptions (WR, TE, RB) | 35,774 | 1.31 vs 1.31 | 0.6% | 0.9% |

"Typical miss" is mean absolute error of the expected value. Brier score and log loss score the over/under probabilities at the lines below; lower is better, and the improvement is relative to the baseline. The baseline is each player's own recent average (same recency weighting) with the same outcome-spread layer, so the comparison isolates what the extra pre-game information adds. [Full summary](outputs/backtest/summary.csv)

### Calibration by line

| Leg | Happened | Model said |
|---|---:|---:|
| Receiving yards over 39.5 | 23.9% | 23.9% |
| Receiving yards over 79.5 | 7.1% | 7.0% |
| Receptions over 3.5 | 24.6% | 24.7% |
| Rushing yards over 49.5 | 25.7% | 25.6% |
| Passing yards over 249.5 | 40.8% | 41.3% |
| Passing yards over 299.5 | 19.7% | 20.7% |
| QB 2+ passing TDs | 45.2% | 44.5% |
| Anytime TD | 19.7% | 19.9% |

[All lines](outputs/backtest/by_line.csv) · [Calibration bands](outputs/backtest/calibration.csv) · [Bias by season](outputs/backtest/season_bias.csv)

For yardage, actual results fell at or below the model's 90th percentile 90% of the time (89.7% for passing yards, 90.1–90.2% for the others). For passing yards, a continuous stat, 10.8% fell at or below the 10th percentile and 50.2% at or below the median.

## How it works

1. **Data.** Weekly box scores for every QB, RB, WR and TE since 2012 and every game's closing spread, total and moneyline, from nflverse ([`fetch_data.py`](fetch_data.py)).
2. **Pre-game features, from earlier games only** ([`features.py`](parlay_model/features.py)):
   - *Usage:* the player's recent share of his team's targets, carries or pass attempts, and the team's recent volume. Recent games count most (exponential weighting, half-life of four games).
   - *Efficiency:* yards per target, carry or attempt, catch rate and touchdown rates over his last 24 games, shrunk toward the position average from earlier seasons, so a few lucky games do not dominate.
   - *Game environment:* the team's implied points (closing total and spread) and no-vig win probability, and what the opponent's defense has allowed recently.
3. **Expected value** ([`marginals.py`](parlay_model/marginals.py)). A Poisson regression per stat. Seasons are weighted so that one three years older counts half as much. [Fitted coefficients](outputs/projections/coefficients.csv): for example, anytime-TD expectations scale almost one-for-one with the team's implied points (coefficient 1.01 on the log scale).
4. **Range of outcomes.** For yards and receptions, the distribution of actual ÷ expected from the latest four seasons, kept separately for ten bands of expected value. This keeps the real shape, including zero-catch games and long tails. Anytime TDs follow a Poisson. Passing TDs vary less than a Poisson allows, so they use a binomial whose number of "chances" is fitted by likelihood.
5. **Backtest** ([`evaluate.py`](parlay_model/evaluate.py)). The model is refit before every four-week block (weeks 1–4, 5–8, 9–12, 13–18) on all earlier games, from 2019 through 2026 week 5, and scored on the block it has not seen.

## This week's projections

[`outputs/projections/2026_week06.csv`](outputs/projections/2026_week06.csv) lists every player expected to play in 2026 week 6: the expected value and the 10th, 25th, 50th, 75th and 90th percentiles for yardage and receptions, and the chance of 1+, 2+ and 3+ touchdowns. Read the percentiles as "a 10% chance of fewer than p10, a 10% chance of more than p90".

**Check injury news before using any of it.** Phase 1 does not read the injury report: a player who appeared in either of his team's last two games is assumed active, and the starting quarterback is the one listed in the schedule.

## Run it

```bash
pip install -r requirements.txt
pytest
python run_backtest.py      # walk-forward backtest, 2019 to the latest week (~1 minute)
python run_projections.py   # projections for the next unplayed week
python fetch_data.py        # optional: refresh the data from nflverse
```

Tests check that features use only earlier games, the shrinkage and EWMA arithmetic by hand, relocated team codes, the upcoming-week roster, that probabilities are calibrated and monotonic on simulated data, and that the model beats the baseline on the real 2025 season.

## What I learned / what I would do differently

The structure helps most where a player's own history is a weak guide. A quarterback's yards and touchdowns depend heavily on the game script the market expects, so implied points and win probability add a lot. A receiver's recent yards already contain most of what usage and efficiency can say, so the gain there is small. I also tried a league-wide passing-environment feature to track seasons like 2023, when passing dropped; it made calibration worse, so I removed it. Next I would add the injury report and snap counts, model quarterback rushing, and, in Phase 2, link the legs.

## Limitations

- **No market prop lines.** These are the model's own probabilities. Without historical prop prices (Phase 4) there is no evidence yet that they beat any market.
- **Injuries and inactives are not modeled.** Projections assume recent players are active.
- **League-wide shifts.** Averaged over a season, passing-yard probabilities ran up to about 4 points high (2021) and 5 points low (2023, when league passing dropped). Passing TDs stayed within 2.5 points by season and the other stats within about 1.2 points.
- **Not yet modeled:** QB rushing, kickers and defense. Fullbacks are grouped with running backs.
- Regular season only. Players need three earlier games to be projected.

## Data

[nflverse](https://github.com/nflverse) schedules and player statistics. The trimmed files in `data/` include a retrieval date, source URLs and SHA-256 hashes.
