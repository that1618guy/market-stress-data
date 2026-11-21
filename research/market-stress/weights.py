"""How sensitive is the Market Stress Index to its weights, ranges and thresholds? (Nov 21, 2026)
The shipped calls it compares against are the equal-thirds calls; the 30/40/30 switch was chosen from this test.
python3 -I research/market-stress/weights.py research/market-stress/data research/market-stress

1. Weight grid: every (fear, fragility, damage) split in 5% steps. Same seals, same washout rule.
2. Out of sample: pick the 'best' weights on one half of history, score them on the other half.
3. Weighting schemes that never look at outcomes: equal, inverse volatility, first principal component.
4. Ranges and thresholds moved one at a time around the shipped values.
Outcomes use closes only; the live Nov 20 session is never an outcome.
"""
import sys, json, itertools
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as we

DATA, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
P = pd.read_csv(DATA / "panel.csv", parse_dates=["date"], index_col="date")
P = P[P.index >= "2003-01-01"].sort_index()
spx = P.spx; LIVE = pd.Timestamp("2026-11-20"); START = pd.Timestamp("2006-03-06"); SPLIT = pd.Timestamp("2016-01-01")
t = we.trend(P)
seal_trend = (t["dist"] < 0).astype(float).where(t["dist"].notna())
idx = list(spx.index)
pos = {d: i for i, d in enumerate(idx)}

def comps(vix=(15, 40), vvix=(90, 150), dmg=0.10, ma=100):
    dist = t["dist"] if ma == 100 else spx / spx.rolling(ma, min_periods=ma).mean() - 1
    f = np.clip((P.vix - vix[0]) / (vix[1] - vix[0]), 0, 1)
    g = np.clip((P.vvix - vvix[0]) / (vvix[1] - vvix[0]), 0, 1)
    d = np.clip(-dist / dmg, 0, 1)
    return f, g, d, dist

def run(w, vix=(15, 40), vvix=(90, 150), dmg=0.10, wmin=80, rv=25.0, ma=100):
    f, g, d, dist = comps(vix, vvix, dmg, ma)
    s = pd.DataFrame(index=P.index)
    s["score"] = 100 * (w[0] * f + w[1] * g + w[2] * d)
    st = (dist < 0).astype(float).where(dist.notna())
    sv = (t["rv20"] > rv).astype(float).where(t["rv20"].notna())
    s["seals"] = st + sv; s["seals_max"] = 2
    o = we.states(s, washout_min=wmin)
    firsts = [x for x in o.index[o.washout_first == 1] if START <= x < LIVE]
    return firsts, s["score"]

def fwd(dt, n):
    i = pos[dt]; j = i + n
    return spx.iloc[j] / spx.iloc[i] - 1 if j < len(spx) and idx[j] < LIVE else np.nan
def dd(dt, n=126):
    i = pos[dt]; win = spx.iloc[i + 1:i + n + 1]; win = win[win.index < LIVE]
    return win.min() / spx.iloc[i] - 1 if len(win) >= 21 else np.nan

BASE, _ = run((1 / 3, 1 / 3, 1 / 3))
def compare(firsts, tol=15):
    """match calls to the shipped (equal-weight) calls within +-tol sessions"""
    matched = [b for b in BASE if any(abs(pos[b] - pos[x]) <= tol for x in firsts)]
    extra = [x for x in firsts if not any(abs(pos[b] - pos[x]) <= tol for b in BASE)]
    shift = [min(abs(pos[b] - pos[x]) for x in firsts) for b in matched]
    return len(matched), extra, (max(shift) if shift else None)

def metrics(firsts, lo=None, hi=None):
    h = [x for x in firsts if (lo is None or x >= lo) and (hi is None or x < hi)]
    r3 = [fwd(x, 63) for x in h]; r12 = [fwd(x, 252) for x in h]; dds = [dd(x) for x in h]
    ok = lambda a: [v for v in a if np.isfinite(v)]
    return {"n": len(h), "dates": [x.strftime("%Y-%m-%d") for x in h],
            "spx_3m": float(np.median(ok(r3))) if ok(r3) else None, "spx_12m": float(np.median(ok(r12))) if ok(r12) else None,
            "up12": sum(1 for v in ok(r12) if v > 0), "n12": len(ok(r12)),
            "dd_med": float(np.median(ok(dds))) if ok(dds) else None, "dd_worst": float(min(ok(dds))) if ok(dds) else None}

# ------------------------------------------------------------------ 1. weight grid
grid = []
for a in range(0, 21):
    for b in range(0, 21 - a):
        c = 20 - a - b
        w = (a / 20, b / 20, c / 20)
        firsts, sc = run(w)
        m = metrics(firsts); k, extra, sh = compare(firsts)
        grid.append({"w": w, **m, "matched": k, "extra": [x.strftime("%Y-%m-%d") for x in extra], "max_shift": sh})
G = pd.DataFrame(grid)
G["same5"] = (G.matched == len(BASE)) & (G.extra.map(len) == 0)

# ------------------------------------------------------------------ 2. out of sample, both directions
def objective(m):
    # reward the 3-month return after entries, penalise the drop still to come; need 2+ calls
    if m["n"] < 2 or m["spx_3m"] is None or m["dd_med"] is None:
        return -9
    return m["spx_3m"] + m["dd_med"]
oos = []
for name, (tr_lo, tr_hi, te_lo, te_hi) in {"fit 2006-2015, test 2016-2026": (START, SPLIT, SPLIT, LIVE),
                                            "fit 2016-2026, test 2006-2015": (SPLIT, LIVE, START, SPLIT)}.items():
    best = None
    for row in grid:
        firsts = [pd.Timestamp(x) for x in row["dates"]]
        j = objective(metrics(firsts, tr_lo, tr_hi))
        if best is None or j > best[0]:
            best = (j, row["w"], firsts)
    oos.append({"split": name, "best_w": best[1],
                "fit_best": metrics(best[2], tr_lo, tr_hi), "fit_equal": metrics(BASE, tr_lo, tr_hi),
                "test_best": metrics(best[2], te_lo, te_hi), "test_equal": metrics(BASE, te_lo, te_hi)})

# ------------------------------------------------------------------ 3. weights that never see outcomes
f, g, d, _ = comps()
X = pd.concat([f, g, d], axis=1).loc[START:LIVE - pd.Timedelta(days=1)].dropna()
iv = 1 / X.std(); iv = tuple((iv / iv.sum()).round(3))
Z = (X - X.mean()) / X.std(); vals, vecs = np.linalg.eigh(np.cov(Z.T.values)); pc = np.abs(vecs[:, -1]); pc = tuple((pc / pc.sum()).round(3))
schemes = []
for name, w in {"equal (shipped)": (1 / 3, 1 / 3, 1 / 3), "inverse volatility": iv, "first principal component": pc}.items():
    firsts, _ = run(w); k, extra, sh = compare(firsts)
    schemes.append({"scheme": name, "w": [round(x, 3) for x in w], **metrics(firsts), "matched": k, "extra": [x.strftime("%Y-%m-%d") for x in extra], "max_shift": sh})
pc_share = float(vals[-1] / vals.sum())

# ------------------------------------------------------------------ 4. ranges and thresholds, one at a time
variants = {"shipped": {}, "VIX 12-35": {"vix": (12, 35)}, "VIX 15-45": {"vix": (15, 45)}, "VIX 20-50": {"vix": (20, 50)},
            "VVIX 80-140": {"vvix": (80, 140)}, "VVIX 85-175 (original's)": {"vvix": (85, 175)}, "VVIX 100-160": {"vvix": (100, 160)},
            "damage full at 8%": {"dmg": 0.08}, "damage full at 12%": {"dmg": 0.12}, "damage full at 15%": {"dmg": 0.15},
            "washout at 75": {"wmin": 75}, "washout at 85": {"wmin": 85},
            "vol line 20%": {"rv": 20.0}, "vol line 30%": {"rv": 30.0},
            "50-day average": {"ma": 50}, "200-day average": {"ma": 200}}
rng = []
for name, kw in variants.items():
    firsts, _ = run((1 / 3, 1 / 3, 1 / 3), **kw); k, extra, sh = compare(firsts)
    rng.append({"variant": name, **metrics(firsts), "matched": k, "extra": [x.strftime("%Y-%m-%d") for x in extra], "max_shift": sh})

# ------------------------------------------------------------------ report
plateau = G[(G.w.map(min) >= 0.2)]
out = {"base": [x.strftime("%Y-%m-%d") for x in BASE], "grid_points": len(G), "same5_all": int(G.same5.sum()),
       "plateau_points": len(plateau), "same5_plateau": int(plateau.same5.sum()),
       "grid": [{**r, "w": list(r["w"])} for r in grid], "oos": oos, "schemes": schemes, "pc1_share": pc_share, "ranges": rng}
(OUT / "weights.json").write_text(json.dumps(out, indent=1, default=float))

pct = lambda x: "  n/a" if x is None else f"{x * 100:+5.1f}%"
print("shipped calls:", out["base"])
print(f"weight grid: {len(G)} splits; same five calls in {int(G.same5.sum())}. With every weight 20%+: {int(plateau.same5.sum())} of {len(plateau)}")
for _, r in plateau.sort_values("w").iterrows():
    print(f"  w={tuple(round(x, 2) for x in r.w)} n={r.n} matched={r.matched} extra={r.extra} shift={r.max_shift} 3m {pct(r.spx_3m)} dd {pct(r.dd_med)}")
print("\nout of sample:")
for o in oos:
    print(" ", o["split"], "best w", o["best_w"])
    for k in ("fit_best", "fit_equal", "test_best", "test_equal"):
        m = o[k]; print(f"    {k:10s} n={m['n']} {m['dates']} 3m {pct(m['spx_3m'])} 12m {pct(m['spx_12m'])} dd {pct(m['dd_med'])}")
print("\nschemes (PC1 explains %.0f%% of the variance):" % (pc_share * 100))
for s in schemes:
    print(f"  {s['scheme']:26s} w={s['w']} n={s['n']} matched={s['matched']} extra={s['extra']} shift={s['max_shift']} 3m {pct(s['spx_3m'])} 12m {pct(s['spx_12m'])} dd {pct(s['dd_med'])} worst {pct(s['dd_worst'])}")
print("\nranges and thresholds:")
for s in rng:
    print(f"  {s['variant']:26s} n={s['n']} matched={s['matched']} extra={s['extra']} shift={s['max_shift']} 3m {pct(s['spx_3m'])} 12m {pct(s['spx_12m'])} ({s['up12']}/{s['n12']}) dd {pct(s['dd_med'])} worst {pct(s['dd_worst'])}")
