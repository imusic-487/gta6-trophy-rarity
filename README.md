# gta6-trophy-rarity

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/imusic-487/gta6-trophy-rarity/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/imusic-487/gta6-trophy-rarity/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A week-one trophy rarity forecaster for make.fun's
`gta6-trophy-rarity-forecast-01` bounty. It reads one case (the house's
reads of Grand Theft Auto VI's PlayStation 5 trophy page on Exophase, up to
the case's checkpoint) and writes, for each of the game's 55 trophies, five
quantiles (5/25/50/75/95%) of its Exophase earn rate at the case's target,
scored by pinball loss on the logit scale.

## Method

Exophase refreshes a game's counts on its own schedule, not continuously:
the page sat unchanged for hours on launch night (638 players tracked),
then jumped to 3,868 at the first refresh, every rate but the 100% one
lower. The last read is a stale snapshot, so persisting it loses — the
house baseline scores about 1.51 on the practice for exactly this reason.
The program models the two moving parts separately and recombines them:

- **Players tracked N(t):** a power law in time since release,
  N(t) = A (1+t)^b, fitted to the distinct reads and anchored at the first
  read with a prior slope b ~ 1.15 (launch night: 638 at release, 3,868 at
  4.5 h, ~12–17k by 11.5 h on the live page). The release-anchored prior
  matters most when the history is flat; the fit takes over once reads move.
- **Earners E_i(t):** a cohort-maturation model. A player who joined at
  time s has, by time t, probability p_i (1 − exp(−(t−s)/tau)) of holding
  trophy i. With N(t) a power law this gives E_i(t) = p_i · G(t; tau),
  where G is one shape per trophy category. tau is fit per category by
  grid search, p_i per trophy by least squares on G. This produces the
  maturation wave (early trophies rise toward their eventual share as the
  player base ages) and the dilution (new tracked players have earned
  nothing yet).
- **Cross-checks blended in logit space:** a shrunk log-time trend of
  logit(rate) (slopes shrunk to the category mean) and an earners power
  law, release-anchored like N(t). Blend weights 0.35 / 0.40 / 0.25.
- **Priors:** a small pull (weight 0.10, scaled by horizon) toward
  week-one prior medians per trophy category (story chapters, missions,
  grind, misc), so a flat history still lands in a plausible range.
- **Spread:** a logit sigma growing with horizon plus the fit residual
  (0.65 floor), so the 90% band covers a refresh jump — one refresh moved
  small rates about 1.5 logit on launch night — as well as trend error.

Trophies are grouped by type from their descriptions (chapter, final
mission, story mission, side strand, grind, activity, fishing, arcade,
misc, plus the platinum, the 100% trophy and the opening mission). No
network, no model calls, standard library only; deterministic output.

## Validation

Practice answers are not published, so the practice score below uses the
published refresh totals (3,868 players, 4,276 trophies earned at 04:30Z)
with each trophy's earners scaled proportionally (408/305); the live-12h
row scores a pseudo-case cut at 03:00Z against the real Exophase page as
fetched 2026-11-19 11:32 UTC. Simulated finals are cohort ground truths
with varied maturation speed, player growth and refresh cadence, plus
structural-break shocks after the deadline (weekend progression surge,
dilution surge). Pinball loss on the logit scale, lower is better.

| set | this entry | house baseline |
| --- | --- | --- |
| practice p00 (proportional earners) | 0.243 | 1.403 |
| practice p01 (proportional earners) | 0.240 | 1.403 |
| practice, no earner growth / fast growth | 0.245 / 0.280 | 1.403 |
| live 12h (real page, 03:00 → 11:32Z) | 0.271 | 1.385 |
| simulated finals S1–S7 | 0.197–0.256 | 0.360–0.553 |
| simulated shocks (surge / dilution) | 0.242–0.634 | — |

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

The program reads every `<id>.in` in the input directory and writes
`<id>.out` (`{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}`)
to the output directory. One file, Python 3.13 standard library only.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
