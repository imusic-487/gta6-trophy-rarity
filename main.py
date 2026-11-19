"""GTA VI trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Forecasts, for each of the game's 55 PlayStation trophies, five quantiles
(5/25/50/75/95%) of its Exophase earn rate at the case's target, from the
house's reads of the trophy page before the case's checkpoint.

Method (see README.md for the full write-up):

  Exophase refreshes a game's counts on its own schedule, not continuously:
  the page can sit unchanged for hours while players keep playing, then jump
  (launch night: 638 players tracked, then 3,868 at the first refresh, every
  rate but the 100% one lower). So the last read is a stale snapshot, and a
  forecaster that just persists it loses. The program models the two moving
  parts separately -- earners and players tracked -- and recombines them:

  1. Players tracked N(t): a power law in time since release,
     N(t) = A (1+t)^b, fitted to the distinct reads and anchored at the
     first read with a prior slope b ~ 1.15 (launch night: 638 -> 3,868 in
     4.5 h -> ~12-17k by 11.5 h). The release-anchored prior matters most
     when the history is flat; the fit takes over once reads move.
  2. Earners E_i(t): a cohort-maturation model. A player who joined at time
     s has, by time t, probability p_i (1 - exp(-(t-s)/tau)) of holding
     trophy i. With N(t) a power law this gives E_i(t) = p_i * G(t; tau),
     where G is the same shape for every trophy in a category. tau is fit
     per category by grid search, p_i per trophy by least squares on G.
     This is the primary median: it produces the maturation wave (early
     trophies rise toward their eventual share as the player base ages)
     and the dilution (new tracked players have earned nothing yet).
  3. Two cross-checks blended in logit space: a shrunk log-time trend of
     logit(rate) (the house curve baseline's shape, fitted on all reads
     with slopes shrunk to the category mean), and an earners power law.
  4. A small pull toward week-one prior medians per trophy category
     (story chapters, missions, grind, misc), scaled by horizon, so a flat
     history still lands in a plausible week-one range.
  5. Spread: a logit sigma growing with horizon plus the fit residual,
     so the 90% band covers a refresh jump (practice: a single refresh
     moved small rates ~1.5 logit) as well as trend error.

Standard library only, no network, deterministic: fixed grids, sorted
file order, no randomness, no wall-clock use.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# --- priors ---------------------------------------------------------------
# Players-tracked power-law slope, from launch night (638 at release,
# 3,868 at 4.5 h, ~12-17k at 11.5 h on the live page).
B_N_PRIOR = 1.15
# Earner power-law slopes by category (used by estimator 3 and as fallback).
D_PRIOR = {
    "first": 1.15, "chapter": 0.9, "final": 0.7, "story": 0.8, "side": 0.7,
    "platinum": 0.6, "hundred": 0.5, "grind": 0.6, "activity": 0.8,
    "fishing": 1.0, "arcade": 0.9, "misc": 0.8,
}
# Maturation timescale (hours) prior by category, for the cohort fit.
TAU_PRIOR = {
    "first": 2.0, "chapter": 48.0, "final": 300.0, "story": 120.0,
    "side": 200.0, "platinum": 600.0, "hundred": 800.0, "grind": 500.0,
    "activity": 150.0, "fishing": 72.0, "arcade": 72.0, "misc": 150.0,
}
# Week-one prior medians (percent) by category. Only a small pull.
PRIOR_MEDIAN = {
    "first": 98.0, "chapter": 15.0, "final": 4.0, "story": 8.0, "side": 3.0,
    "platinum": 0.30, "hundred": 0.20, "grind": 0.8, "activity": 4.0,
    "fishing": 10.0, "arcade": 8.0, "misc": 4.0,
}
CHAPTER_PRIOR = {1: 25.0, 2: 15.0, 3: 10.0, 4: 7.0}
GRIND_PRIOR = {"vehicles": 3.0, "explore": 1.5}
TAU_GRID = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0]

# Blend weights (logit space) and prior pull; tuned on practice + simulated finals.
W_COHORT = 0.35
W_TREND = 0.40
W_EARN = 0.25
W_PRIOR = 0.10
# Spread: sigma = SIGMA_MULT * sqrt((SIGMA_BASE0 + SIGMA_BASE1*sqrt(days))^2 + resid^2)
SIGMA_MULT = 1.0
SIGMA_BASE0 = 0.35
SIGMA_BASE1 = 0.25
SIGMA_FLOOR = 0.65
SIGMA_CAP = 2.0


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-max(-30.0, min(30.0, lg))))


def round4(x):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_release(dt):
    return max(0.0, (dt - RELEASE).total_seconds() / 3600.0)


def category(trophy):
    tid = trophy["id"]
    desc = trophy["description"]
    tier = trophy["tier"]
    if tid == 1 or tier == "platinum":
        return "platinum"
    if tid == 2:
        return "first"
    if tid == 14:
        return "hundred"
    if desc.startswith("Complete Chapter"):
        return "chapter"
    if desc == "Complete the final mission.":
        return "final"
    if "final mission for" in desc or "mission strands for strangers" in desc:
        return "side"
    if desc.startswith("Complete ") and tier == "bronze" and "Complete the" not in desc:
        return "story"
    if any(k in desc for k in ("250 Stunts", "Collectible Card sets",
                               "100 locations", "50 Challenges")):
        return "grind"
    if "Discover 100 vehicles" in desc or "Explore all of Leonida" in desc:
        return "grind"
    if any(k in desc for k in ("Try each type of activity", "gold medal",
                               "Drone Race", "different races")):
        return "activity"
    if "Beat Cal" in desc:
        return "fishing"
    if "arcade game" in desc:
        return "arcade"
    return "misc"


def chapter_no(trophy):
    try:
        return int(trophy["description"].rsplit(" ", 1)[1].rstrip("."))
    except Exception:
        return 0


def prior_median(trophy):
    cat = category(trophy)
    if cat == "chapter":
        return CHAPTER_PRIOR.get(chapter_no(trophy), 15.0)
    if cat == "grind":
        if "Discover 100 vehicles" in trophy["description"]:
            return GRIND_PRIOR["vehicles"]
        if "Explore all of Leonida" in trophy["description"]:
            return GRIND_PRIOR["explore"]
    return PRIOR_MEDIAN[cat]


def ols(xs, ys):
    """Least squares y = a + b x. Returns (a, b)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        return ys[0], 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return my, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - b * mx, b


def distinct_states(case):
    """Collapse consecutive identical reads to their first occurrence."""
    out = []
    prev = None
    for h in case["history"]:
        key = (h["players_tracked"], tuple(h["rates"]))
        if key != prev:
            out.append(h)
            prev = key
    return out


def fit_n_model(case, states, t_star):
    """Power law N(t) = A (1+t)^b. Returns log N at t_star and at any t."""
    t0 = hours_since_release(parse_time(case["history"][0]["at"]))
    n_first = case["history"][0]["players_tracked"]
    xs = [math.log1p(hours_since_release(parse_time(h["at"]))) for h in states]
    ys = [math.log(max(1, h["players_tracked"])) for h in states]
    a_ols, b_ols = ols(xs, ys)
    # Release-anchored prior: N(t) = N_first * ((1+t)/(1+t0))^B_N_PRIOR.
    x0 = math.log1p(t0)
    log_n_first = math.log(max(1, n_first))

    def log_n_prior(t):
        return log_n_first + B_N_PRIOR * (math.log1p(t) - x0)

    def log_n_ols(t):
        return a_ols + b_ols * math.log1p(t)

    # Weight the fit by how much the page actually moved.
    w = 0.0
    if len(states) >= 3:
        w = min(0.75, 0.15 * (len(states) - 1))
    b_ols_c = max(0.0, min(3.0, b_ols))
    # Re-anchor the OLS line at the last distinct state so its level is fresh.
    t_last = hours_since_release(parse_time(states[-1]["at"])) if states else 0.0
    n_last = max(1, states[-1]["players_tracked"]) if states else 1
    log_n_last = math.log(n_last)
    x_last = math.log1p(t_last)
    if states:
        log_n_star = w * (log_n_last + b_ols_c * (math.log1p(t_star) - x_last)) \
            + (1 - w) * log_n_prior(t_star)
    else:
        log_n_star = log_n_prior(t_star)

    def log_n(t):
        v_ols = log_n_last + b_ols_c * (math.log1p(t) - x_last) if states \
            else a_ols + b_ols_c * math.log1p(t)
        return w * v_ols + (1 - w) * log_n_prior(t)

    return log_n_star, log_n


def g_shape(t, tau, log_n):
    """G(t; tau) = E earners with p=1: integral of n(s)(1-exp(-(t-s)/tau)) ds.

    n(s) comes from the N power law: n(s) = d/ds exp(log_n(s)), computed
    numerically on a coarse age grid. t in hours.
    """
    if t <= 0:
        return 0.0
    steps = 48
    dt = t / steps
    total = 0.0
    prev_n = math.exp(log_n(0.0))
    for k in range(1, steps + 1):
        s = k * dt
        n_s = math.exp(log_n(s))
        # players joining in (s-dt, s]: n(s)-n(s-dt) approx by N difference
        joined = max(0.0, n_s - prev_n)
        age = t - (s - dt / 2.0)
        f = 1.0 - math.exp(-max(0.0, age) / tau)
        total += joined * f
        prev_n = n_s
    return total


def forecast_case(case):
    trophies = case["trophies"]
    levels = case["levels"]
    assert levels == LEVELS
    horizon_days = case["horizon_minutes"] / 1440.0
    t_star = hours_since_release(parse_time(case["target"]))
    cats = [category(t) for t in trophies]
    states = distinct_states(case)
    if not states:
        raise ValueError("case has no history")
    last = case["history"][-1]
    t_last = hours_since_release(parse_time(last["at"]))
    x_star = math.log1p(t_star)
    x_last = math.log1p(t_last)

    log_n_star, log_n = fit_n_model(case, states, t_star)
    n_star = math.exp(log_n_star)

    # Cache G(t; tau): the same (t, tau) pairs recur across trophies.
    _g_cache = {}

    def G(t, tau):
        key = (round(t, 3), tau)
        if key not in _g_cache:
            _g_cache[key] = g_shape(t, tau, log_n)
        return _g_cache[key]

    # Per-state times and earners.
    state_t = [hours_since_release(parse_time(h["at"])) for h in states]
    state_n = [max(1, h["players_tracked"]) for h in states]
    earners = []  # earners[i][j] or None
    for i in range(len(trophies)):
        row = []
        for h in states:
            r = h["rates"][i]
            row.append(None if r is None else r * h["players_tracked"] / 100.0)
        earners.append(row)

    # --- cohort fit per category: tau by grid, p_i by least squares on G ---
    cat_tau = {}
    cat_p = {}
    cat_resid = {}
    for cat in sorted(set(cats)):
        idxs = [i for i, c in enumerate(cats) if c == cat]
        # Pooled earners for tau selection (only states with all present).
        pooled_t, pooled_e = [], []
        for j, t in enumerate(state_t):
            vals = [earners[i][j] for i in idxs]
            if all(v is not None for v in vals):
                pooled_t.append(t)
                pooled_e.append(sum(vals))
        best = None
        tau_use = TAU_PRIOR[cat]
        if len(pooled_t) >= 2 and sum(pooled_e) > 0:
            for tau in TAU_GRID:
                gs = [G(t, tau) for t in pooled_t]
                denom = sum(g * g for g in gs)
                if denom <= 0:
                    continue
                p_hat = sum(g * e for g, e in zip(gs, pooled_e)) / denom
                if p_hat <= 0:
                    continue
                sse = sum((e - p_hat * g) ** 2 for g, e in zip(gs, pooled_e))
                scale = sum(e * e for e in pooled_e) or 1.0
                rel = sse / scale
                if best is None or rel < best[0]:
                    best = (rel, tau, p_hat)
            if best is not None:
                tau_use = best[1]
        cat_tau[cat] = tau_use
        # Per-trophy p_i by least squares of E_i on G over states.
        ps = {}
        for i in idxs:
            gs, es = [], []
            for j, t in enumerate(state_t):
                if earners[i][j] is None:
                    continue
                g = G(t, tau_use)
                if g > 0.5:
                    gs.append(g)
                    es.append(earners[i][j])
            if gs:
                denom = sum(g * g for g in gs)
                p_i = sum(g * e for g, e in zip(gs, es)) / denom if denom > 0 else 0.0
            else:
                r_last = last["rates"][i]
                p_i = (r_last / 100.0) if r_last is not None else 0.0
            ps[i] = max(0.0, min(0.995, p_i))
        cat_p[cat] = ps
        # Residual (logit scale) for the spread: compare last-state fit.
        res = []
        for i in idxs:
            if last["rates"][i] is None:
                continue
            e_fit = ps[i] * G(t_last, tau_use)
            r_fit = 100.0 * e_fit / max(1.0, state_n[-1])
            res.append(abs(logit(last["rates"][i]) - logit(r_fit)))
        cat_resid[cat] = (sum(res) / len(res)) if res else 0.3

    # --- category trend slopes (logit rate vs log time), shrunk ---
    cat_slope = {}
    for cat in sorted(set(cats)):
        slopes = []
        for i, c in enumerate(cats):
            if c != cat:
                continue
            xs, ys = [], []
            for j, t in enumerate(state_t):
                r = states[j]["rates"][i]
                if r is not None and 0 < r < 100:
                    xs.append(math.log1p(t))
                    ys.append(logit(r))
            if len(xs) >= 2:
                _, b = ols(xs, ys)
                slopes.append(max(-2.0, min(3.0, b)))
        slopes.sort()
        cat_slope[cat] = slopes[len(slopes) // 2] if slopes else 0.0

    rows = []
    for i, trophy in enumerate(trophies):
        cat = cats[i]
        r_last = last["rates"][i]
        if r_last is None:
            # Fall back to the most recent non-null read.
            r_last = 0.5
            for h in reversed(case["history"]):
                if h["rates"][i] is not None:
                    r_last = h["rates"][i]
                    break
        lg_last = logit(r_last)

        # Estimator 1: cohort.
        g_star = G(t_star, cat_tau[cat])
        e_cohort = cat_p[cat][i] * g_star
        rate_cohort = 100.0 * e_cohort / max(1.0, n_star)
        lg_cohort = logit(rate_cohort)

        # Estimator 2: shrunk log-time trend from the last rate.
        xs, ys = [], []
        for j, t in enumerate(state_t):
            r = states[j]["rates"][i]
            if r is not None and 0 < r < 100:
                xs.append(math.log1p(t))
                ys.append(logit(r))
        if len(xs) >= 2:
            _, b_i = ols(xs, ys)
            b_i = max(-2.0, min(3.0, b_i))
            n_eff = len(xs)
            b_shrunk = (n_eff * b_i + 3.0 * cat_slope[cat]) / (n_eff + 3.0)
        else:
            b_shrunk = cat_slope[cat]
        lg_trend = lg_last + b_shrunk * (x_star - x_last)
        lg_trend = max(lg_last - 3.0, min(lg_last + 3.5, lg_trend))

        # Estimator 3: earners power law, release-anchored.
        e_first = None
        for h in case["history"]:
            if h["rates"][i] is not None:
                e_first = h["rates"][i] * h["players_tracked"] / 100.0
                t_first = hours_since_release(parse_time(h["at"]))
                break
        if e_first is None or e_first <= 0:
            lg_earn = lg_cohort
        else:
            d = D_PRIOR[cat]
            if len(xs) >= 2:
                # Fit log E on log time, shrink to the prior slope.
                exs, eys = [], []
                for j, t in enumerate(state_t):
                    if earners[i][j] is not None and earners[i][j] > 0:
                        exs.append(math.log1p(t))
                        eys.append(math.log(earners[i][j]))
                if len(exs) >= 2:
                    _, b_e = ols(exs, eys)
                    d = (len(exs) * max(0.0, min(2.5, b_e)) + 2.0 * d) / (len(exs) + 2.0)
            e_star = e_first * ((1.0 + t_star) / (1.0 + t_first)) ** d
            lg_earn = logit(100.0 * e_star / max(1.0, n_star))

        lg_med = W_COHORT * lg_cohort + W_TREND * lg_trend + W_EARN * lg_earn
        # Small pull toward the week-one category prior, scaled by horizon.
        w_prior = W_PRIOR * min(1.0, horizon_days / 4.52)
        lg_med = (1 - w_prior) * lg_med + w_prior * logit(prior_median(trophy))

        # Spread: horizon base + category fit residual.
        base = SIGMA_BASE0 + SIGMA_BASE1 * math.sqrt(max(0.0, horizon_days))
        resid = max(0.10, min(0.70, cat_resid[cat]))
        sigma = max(SIGMA_FLOOR, min(SIGMA_CAP, SIGMA_MULT * math.sqrt(base * base + resid * resid)))

        qs = []
        for z in Z:
            v = inv_logit(lg_med + z * sigma)
            v = round4(min(99.99, max(0.01, v)))
            qs.append(v)
        # Keep monotone after rounding.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
