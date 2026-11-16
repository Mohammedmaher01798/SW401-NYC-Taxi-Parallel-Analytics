"""Parse results/*.log, compute T_P (median), speedup, efficiency, serial fraction; write plots + CSV tables."""
import os, re, csv
from collections import defaultdict
from statistics import median
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results"); OUT = os.path.join(ROOT, "plots")
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": .3, "figure.dpi": 130,
                     "axes.spines.top": False, "axes.spines.right": False})

def lines(name):
    raw = open(os.path.join(RES, name), "rb").read()   # Tee-Object in Windows PowerShell writes UTF-16
    txt = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8", "ignore")
    return [l.strip() for l in txt.splitlines() if l.strip()]

def seq_median(name):
    for l in lines(name):
        if l.startswith("SEQ_MEDIAN"): return float(l.split(",")[1])

def scaling(name):
    d = defaultdict(list)
    for l in lines(name):
        if l.startswith("SCALE,"):
            p = l.split(","); d[int(p[1])].append(float(p[3]))
    return d

rows = []
fig_data = {}
for tag, seqlog, sclog in [("full", "seq_full.log", "omp_scaling_full.log"),
                           ("500mb", "seq_500mb.log", "omp_scaling_500mb.log"),
                           ("100mb", "seq_100mb.log", "omp_scaling_100mb.log")]:
    ts = seq_median(seqlog); sc = scaling(sclog)
    res = []
    for p in (1, 2, 4, 8):
        tp = median(sc[p]); S = ts / tp; E = S / p
        f = (1 / S - 1 / p) / (1 - 1 / p) if p > 1 else float("nan")
        res.append((p, tp, S, E, f))
        rows.append([tag, p, ts, tp, S, E, f, " ".join(f"{x:.4f}" for x in sc[p])])
    fig_data[tag] = (ts, res)
with open(os.path.join(OUT, "benchmark_summary.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["dataset", "threads", "T_S_s", "T_P_median_s", "speedup", "efficiency", "serial_fraction", "all_runs_s"])
    for r in rows: w.writerow([r[0], r[1], f"{r[2]:.6f}", f"{r[3]:.6f}", f"{r[4]:.4f}", f"{r[5]:.4f}", f"{r[6]:.4f}", r[7]])

ps = [1, 2, 4, 8]
# ---- Speedup -----
fig, ax = plt.subplots(figsize=(6, 4.2))
ax.plot(ps, ps, "k--", label="Ideal (S = p)")
for tag, mk in [("full", "o-"), ("500mb", "s-"), ("100mb", "^-")]:
    ax.plot(ps, [r[2] for r in fig_data[tag][1]], mk, label={"full": "Full (clean_full.csv)", "500mb": "500 MB sample", "100mb": "100 MB sample"}[tag])
ax.axvline(4, color="gray", ls=":", lw=1); ax.text(4.1, 6.2, "4 physical cores\n(8 logical via HT)", fontsize=8, color="gray")
ax.set_xticks(ps); ax.set_xlabel("Threads (p)"); ax.set_ylabel("Speedup S = T_S / T_P"); ax.set_title("OpenMP speedup vs. threads")
ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(OUT, "speedup_vs_threads.png")); plt.close(fig)
# ---- Efficiency -----
fig, ax = plt.subplots(figsize=(6, 4.2))
ax.axhline(1, color="k", ls="--", label="Ideal (E = 1)")
for tag, mk in [("full", "o-"), ("500mb", "s-"), ("100mb", "^-")]:
    ax.plot(ps, [r[3] for r in fig_data[tag][1]], mk, label={"full": "Full", "500mb": "500 MB", "100mb": "100 MB"}[tag])
ax.set_xticks(ps); ax.set_ylim(0, 1.15); ax.set_xlabel("Threads (p)"); ax.set_ylabel("Efficiency E = S / p"); ax.set_title("OpenMP parallel efficiency")
ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(OUT, "efficiency_vs_threads.png")); plt.close(fig)

# ---- Scheduling -----
sd = defaultdict(list)
for l in lines("omp_sched_full.log"):
    if l.startswith("SCHED,"):
        p = l.split(","); sd[(p[1], int(p[2]))].append(float(p[5]))
labels, vals, cols = [], [], []
cmap = {"static": "#1f77b4", "dynamic": "#d62728", "guided": "#2ca02c"}
for (k, c), v in sd.items():
    labels.append(f"{k}\n{'default' if c == 0 else c}"); vals.append(median(v)); cols.append(cmap[k])
with open(os.path.join(OUT, "schedule_summary.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["schedule", "chunk", "median_s"])
    for (k, c), v in sd.items(): w.writerow([k, c, f"{median(v):.6f}"])
fig, ax = plt.subplots(figsize=(8.5, 4.2))
bars = ax.bar(range(len(vals)), vals, color=cols)
ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Median time (s), 8 threads"); ax.set_title("Scheduling policy comparison (static / dynamic / guided; x-axis = chunk size)")
for b, v in zip(bars, vals): ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in cmap.values()], labels=list(cmap)); ax.grid(axis="x", alpha=0)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "scheduling_comparison.png")); plt.close(fig)

# ---- Sync strategies + cache -----
sy = defaultdict(list)
for l in lines("omp_sync_100mb.log"):
    if l.startswith("SYNC,"):
        p = l.split(","); sy[p[1]].append(float(p[4]))
ca = defaultdict(dict); cache_rows = []
for l in lines("omp_cache_full.log"):
    if l.startswith("CACHE,"):
        p = l.split(","); cache_rows.append(p[1:])
        ca[p[1]][(p[2], p[3])] = float(p[4])
with open(os.path.join(OUT, "sync_cache_summary.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["experiment", "variant", "config", "median_s"])
    for k, v in sy.items(): w.writerow(["sync_8thr_100mb", k, "", f"{median(v):.6f}"])
    for r in cache_rows: w.writerow(["cache"] + r)

fig, axs = plt.subplots(1, 3, figsize=(13, 4))
st = [(int(k[0]), v) for k, v in ca["stride"].items() if k[1] == "serial"]
st.sort()
axs[0].plot([s for s, _ in st], [v for _, v in st], "o-", color="#d62728", label="serial")
st8 = sorted((int(k[0]), v) for k, v in ca["stride"].items() if k[1] != "serial")
axs[0].plot([s for s, _ in st8], [v for _, v in st8], "s-", color="#1f77b4", label="OpenMP 8 thr")
axs[0].set_xscale("log", base=2); axs[0].set_yscale("log"); axs[0].set_xlabel("Stride (floats; 16 = 1 cache line)")
axs[0].set_ylabel("Time to read all elements (s)"); axs[0].set_title("Stride vs. contiguous access"); axs[0].legend()
names = [("layout", "AoS_24B", "serial", "AoS\n(serial)"), ("layout", "SoA_6B", "serial", "SoA\n(serial)"),
         ("layout", "AoS_24B", "omp8", "AoS\n(8 thr)"), ("layout", "SoA_6B", "omp8", "SoA\n(8 thr)")]
vals_b = [ca[n[0]][(n[1], n[2])] for n in names]
bars_b = axs[1].bar([n[3] for n in names], vals_b, color=["#ff7f0e", "#2ca02c", "#ff7f0e", "#2ca02c"])
for b, val in zip(bars_b, vals_b): axs[1].text(b.get_x() + b.get_width()/2, val, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
axs[1].set_ylabel("Median time (s)"); axs[1].set_title("Data layout: AoS (24 B) vs. SoA (6 B)"); axs[1].tick_params(axis="x", labelsize=8)
fs = [("adjacent_counters", "adjacent\n(false sharing)"), ("padded_counters", "padded\n(64B apart)"), ("reduction_clause", "reduction\nclause")]
axs[2].bar([f[1] for f in fs], [ca["falsesharing"][(f[0], "omp8")] for f in fs], color=["#d62728", "#1f77b4", "#2ca02c"])
axs[2].set_ylabel("Time (s), 8 threads"); axs[2].set_title("False sharing"); axs[2].tick_params(axis="x", labelsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "cache_experiments.png")); plt.close(fig)

fig, ax = plt.subplots(figsize=(5.5, 3.8))
order = ["thread_local_arrays", "atomic_shared", "critical_shared"]
v = [median(sy[k]) for k in order]
ax.bar(["thread-local\narrays", "atomic\n(shared)", "critical\n(shared)"], v, color=["#2ca02c", "#ff7f0e", "#d62728"])
ax.set_yscale("log"); ax.set_ylabel("Median time (s), 8 threads, 100 MB"); ax.set_title("Synchronisation strategy")
for i, x in enumerate(v): ax.text(i, x, f"{x:.3f}", ha="center", va="bottom", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "sync_comparison.png")); plt.close(fig)
print("plots written to", OUT)
for r in rows: print(r[:7])
