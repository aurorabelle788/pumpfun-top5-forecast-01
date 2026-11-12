# pumpfun-top5-forecast-01 entry for @dirtytesla (make.fun).
#
# Strategy: a principled, model-free Monte Carlo forecaster, standard library
# only, fully deterministic (seeded RNG, no wall clock, no network).
#
# Each coin's log10 market cap (in SOL) is modelled as a drifted random walk
# from the cut:  L(h) ~ Normal(L0 + mu * h, sqrt(sigma_rw^2 * h + sigma0^2)).
# mu is a shrunk share of the coin's own recent slope relative to the
# market; sigma is the empirical 10-minute return volatility of the top
# coins, so the walk is as jumpy as the market really is. On top of the
# per-coin walk, every simulation draws one common market shock: the whole
# top-5 cap scale drifts and wobbles together, and that common mode is what
# makes a place's size distribution honest (a per-coin walk alone leaves
# the order statistics far too tight). For each of the seven answer reads (horizons from
# task.json) we simulate the joint ranking of every coin the last mark
# lists, and average the place probabilities and the place-size distributions
# over the reads: the score is the mean of the per-read log scores, so the
# mixture over reads is the honest forecast. A per-place newcomer mass,
# measured from the history's own churn (the share of top-5 slots taken by
# coins absent from the previous mark) and growing with horizon, is left
# unnamed for coins born after the cut: the scorer shares it over the
# answer chart's unnamed coins.
#
# Input:  $MAKEFUN_INPUT_DIR (else /input): task.json, history.ndjson
# Output: $MAKEFUN_OUTPUT_DIR (else /output): forecast.json

import json
import math
import os
import random
from datetime import datetime

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

SEED = 20261112
SIMS_PER_READ = 3000

# --- calibration (measured from the history at run time where possible) ---
DRIFT_SHRINK = 0.30      # share of a coin's own slope over the market drift
DRIFT_CAP = 0.60         # log10 per hour, either way
SIGMA_CAP = 0.80         # log10 per sqrt-hour
SIGMA0 = 0.10            # log10: fixed model error at any horizon
NEWCOMER_CAP = [0.60, 0.65, 0.70, 0.75, 0.80]  # per place 1..5
SIZE_FLOOR = 0.0002      # per bin, smoothed in before renormalising
PROB_FLOOR = 0.0002      # per (coin, place) row we write


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def load_task(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_marks(path):
    marks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m.get("ok") is True and isinstance(m.get("rows"), list):
                marks.append(m)
    if not marks:
        raise ValueError("the history holds no ok mark")
    return marks


def logcap_sol(row):
    return math.log10(int(row["market_cap_lamports"])) - 9.0


def linregress(xs, ys):
    """Least-squares slope and residual sd; returns (slope, resid_sd)."""
    n = len(xs)
    if n < 2:
        return 0.0, None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return 0.0, None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    if n < 3:
        return slope, None
    sse = sum((y - (my + slope * (x - mx))) ** 2 for x, y in zip(xs, ys))
    return slope, math.sqrt(sse / (n - 2))


def floor6(p):
    """Round down to 6 decimals: sums of floored values never exceed the
    unrounded sum, so a place's probabilities stay within its cap."""
    return math.floor(p * 1000000.0) / 1000000.0


def bin_of(x, edges):
    if x < edges[0]:
        return 0
    for i in range(1, len(edges)):
        if x < edges[i]:
            return i
    return len(edges)


def median(vals):
    v = sorted(vals)
    return v[len(v) // 2] if v else 0.0


def main():
    task = load_task(os.path.join(INPUT_DIR, "task.json"))
    marks = load_marks(os.path.join(INPUT_DIR, "history.ndjson"))
    places = int(task["places"])
    edges = [float(e) for e in task["size"]["edges"]]
    n_bins = len(edges) + 1

    times = [parse_time(m["at"]) for m in marks]
    cut = times[-1]
    t0 = times[0]
    horizons = []
    for r in task["reads"]:
        h = (parse_time(r) - cut).total_seconds() / 3600.0
        horizons.append(max(h, 0.0))
    mean_h = sum(horizons) / len(horizons) if horizons else 0.0

    # --- per-coin time series of log10 cap (hours since the first mark) ---
    series = {}
    for m, t in zip(marks, times):
        th = (t - t0).total_seconds() / 3600.0
        for row in m["rows"]:
            series.setdefault(row["mint"], []).append((th, logcap_sol(row)))
    last_rows = sorted(marks[-1]["rows"], key=lambda r: r["rank"])
    last_mints = [r["mint"] for r in last_rows]

    # --- empirical 10-minute return volatility of the top coins ---
    # Pooled sd of consecutive-mark log returns (marks are ~10 min apart).
    # This is the market's real jumpiness, and the walk's sigma per sqrt-hour.
    rets = []
    for i in range(len(marks) - 1):
        prev = {r["mint"]: logcap_sol(r) for r in marks[i]["rows"]}
        for r in marks[i + 1]["rows"]:
            if r["mint"] in prev:
                rets.append(logcap_sol(r) - prev[r["mint"]])
    if len(rets) >= 20:
        mean_r = sum(rets) / len(rets)
        var_r = sum((x - mean_r) ** 2 for x in rets) / len(rets)
        sigma_rw = min(SIGMA_CAP, math.sqrt(var_r) * math.sqrt(6.0))
    else:
        sigma_rw = 0.45
    sigma_rw = max(0.15, sigma_rw)

    # --- market drift: median per-hour change of the top-10 coins' caps
    # over the last hour of history (shrunk halfway toward zero) ---
    market_slopes = []
    for i in range(len(marks) - 1):
        prev = {r["mint"]: logcap_sol(r) for r in marks[i]["rows"]}
        dt = (times[i + 1] - times[i]).total_seconds() / 3600.0
        if dt <= 0:
            continue
        for r in sorted(marks[i + 1]["rows"], key=lambda x: x["rank"])[:10]:
            if r["mint"] in prev:
                market_slopes.append((logcap_sol(r) - prev[r["mint"]]) / dt)
    market_drift = 0.5 * median(market_slopes) if market_slopes else 0.0
    market_drift = max(-DRIFT_CAP, min(DRIFT_CAP, market_drift))

    # --- market scale: the log of the median top-5 cap, its drift and its
    # 10-minute volatility (the common shock that moves every place) ---
    scale_logs = []
    for m in marks:
        top5 = sorted(m["rows"], key=lambda r: r["rank"])[:places]
        if len(top5) == places:
            scale_logs.append(median([logcap_sol(r) for r in top5]))
    scale_changes = [scale_logs[i + 1] - scale_logs[i] for i in range(len(scale_logs) - 1)]
    if len(scale_changes) >= 2:
        mean_sc = sum(scale_changes) / len(scale_changes)
        var_sc = sum((x - mean_sc) ** 2 for x in scale_changes) / len(scale_changes)
        scale_vol = min(0.80, max(0.20, math.sqrt(var_sc) * math.sqrt(6.0)))
        scale_drift = max(-0.50, min(0.50, 0.5 * mean_sc * 6.0))
    else:
        scale_vol, scale_drift = 0.40, 0.0

    # --- newcomer rate: share of top-5 slots taken by coins absent from the
    # previous mark's rows, over the last two transitions (the mature end) ---
    newcomer_rates = []
    for i in range(1, len(marks)):
        prev_rows = marks[i - 1]["rows"]
        if len(prev_rows) < 50:
            continue  # an empty or near-empty mark says nothing about churn
        prev = {r["mint"] for r in prev_rows}
        cur5 = sorted(marks[i]["rows"], key=lambda r: r["rank"])[:places]
        newcomer_rates.append(sum(1 for r in cur5 if r["mint"] not in prev) / places)
    newcomer_rates = newcomer_rates[-3:]  # the mature end of the history
    r10 = min(0.5, median(newcomer_rates)) if newcomer_rates else 0.2

    # --- per-candidate drift and volatility ---
    cand = []
    for mint in last_mints:
        pts = series.get(mint, [])
        l0 = pts[-1][1] if pts else 0.0
        recent = [(x, y) for (x, y) in pts if pts[-1][0] - x <= 1.0] if pts else []
        if len(recent) >= 2:
            slope, _sd = linregress([p[0] for p in recent], [p[1] for p in recent])
        elif len(pts) >= 2:
            slope, _sd = linregress([p[0] for p in pts], [p[1] for p in pts])
        else:
            slope = 0.0
        n_pts = len(pts)
        shrink = n_pts / (n_pts + 2.0)
        mu = DRIFT_SHRINK * shrink * (slope - market_drift)
        mu = max(-DRIFT_CAP, min(DRIFT_CAP, mu))
        cand.append((mint, l0, mu))
    cand.sort(key=lambda c: c[1], reverse=True)

    # --- Monte Carlo over the answer reads ---
    rng = random.Random(SEED)
    place_counts = [[0] * len(cand) for _ in range(places)]
    size_counts = [[0] * n_bins for _ in range(places)]
    total_sims = 0
    for h in horizons:
        means = [l0 + mu * h for (_, l0, mu) in cand]
        sd = math.sqrt(sigma_rw * sigma_rw * h + SIGMA0 * SIGMA0)
        shock_sd = scale_vol * math.sqrt(h) if h > 0 else 0.0
        shock_mean = scale_drift * h
        for _ in range(SIMS_PER_READ):
            shock = rng.gauss(shock_mean, shock_sd) if shock_sd > 0 else shock_mean
            draws = [(rng.gauss(means[j], sd) + shock, j) for j in range(len(cand))]
            draws.sort(key=lambda d: d[0], reverse=True)
            total_sims += 1
            for k in range(places):
                if k < len(draws):
                    place_counts[k][draws[k][1]] += 1
                    size_counts[k][bin_of(draws[k][0], edges)] += 1

    # --- newcomer mass per place (grows with the mean horizon, saturates) ---
    newcomer = []
    for k in range(places):
        steps = 6.0 * mean_h  # 10-minute steps in the mean horizon
        q = NEWCOMER_CAP[k] * (1.0 - (1.0 - r10) ** steps) if steps > 0 else 0.0
        newcomer.append(min(q, NEWCOMER_CAP[k]))

    # --- assemble the forecast ---
    probabilities = []
    for k in range(places):
        q = newcomer[k]
        rows = []
        named_sum = 0.0
        for j, (mint, _, _) in enumerate(cand):
            p = (1.0 - q) * place_counts[k][j] / total_sims if total_sims else 0.0
            if p >= PROB_FLOOR:
                rows.append((mint, p))
                named_sum += p
        if rows and named_sum > 0:
            scale = (1.0 - q) / named_sum
            for mint, p in rows:
                probabilities.append({"mint": mint, "place": k + 1, "p": floor6(p * scale)})
    weight = {}
    for k in range(places):
        for j, (mint, _, _) in enumerate(cand):
            weight[mint] = weight.get(mint, 0.0) + place_counts[k][j] / total_sims * (places - k)
    top5 = sorted(weight, key=lambda m: (-weight[m], m))[:places]
    if len(top5) < places:
        top5 = (top5 + last_mints)[:places]

    sizes = []
    for k in range(places):
        counts = size_counts[k]
        tot = float(sum(counts))
        if tot <= 0:
            continue
        row = []
        for b in range(n_bins):
            p = counts[b] / tot
            p = p * (1.0 - n_bins * SIZE_FLOOR) + SIZE_FLOOR
            if p >= 0.0005:
                row.append((b, floor6(p)))
        s = sum(p for _, p in row)
        if s > 1.0:
            row = [(b, floor6(p / s)) for b, p in row]
        for b, p in row:
            sizes.append({"place": k + 1, "bin": b, "p": p})

    forecast = {"probabilities": probabilities, "top5": top5, "sizes": sizes}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, task.get("output", "forecast.json"))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(forecast, sort_keys=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
