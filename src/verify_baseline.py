"""Independent correctness check of the C++ baseline: recompute the aggregates with pandas on a CSV
and compare with results/seq_results_<x>.csv written by sequential.exe. (Verification only.)
Usage: python src/verify_baseline.py data/sample_100mb.csv results/seq_results_100mb.csv"""
import sys, numpy as np, pandas as pd
src, res = sys.argv[1], sys.argv[2]
d = pd.read_csv(src)
d["tippct"] = 100 * d.tip_amount / d.fare_amount
d["fpm"] = d.fare_amount / d.trip_distance
g = d.groupby(["PULocationID", "pickup_hour"])
ref = pd.DataFrame({"trips": g.size(), "avg_distance": g.trip_distance.mean(), "total_fare": g.fare_amount.sum(),
                    "total_revenue": g.total_amount.sum(), "avg_tip_pct": g.tippct.mean(), "avg_fare_per_mile": g.fpm.mean()})
m, s = g.fpm.transform("mean"), g.fpm.transform(lambda x: x.std(ddof=0))
cnt = g.fpm.transform("size")
d["anom"] = (cnt >= 30) & (d.fpm > m + 3 * s)
ref["anomalies"] = d.groupby(["PULocationID", "pickup_hour"]).anom.sum().astype(int)
ref = ref.reset_index().rename(columns={"PULocationID": "zone", "pickup_hour": "hour"})
out = pd.read_csv(res)
mg = ref.merge(out, on=["zone", "hour"], suffixes=("_py", "_cpp"))
assert len(mg) == len(ref) == len(out), (len(mg), len(ref), len(out))
ok = True
for c in ["trips", "avg_distance", "total_fare", "total_revenue", "avg_tip_pct", "avg_fare_per_mile"]:
    good = np.allclose(mg[c + "_py"], mg[c + "_cpp"], rtol=1e-4, atol=0.01)
    ok &= good; print(f"{c:18s} {'MATCH' if good else 'MISMATCH'}  max_rel_diff={np.max(np.abs(mg[c+'_py']-mg[c+'_cpp'])/np.maximum(1,np.abs(mg[c+'_py']))):.2e}")
da = int(np.abs(mg.anomalies_py - mg.anomalies_cpp).sum())
print(f"anomalies          total_py={mg.anomalies_py.sum()} total_cpp={mg.anomalies_cpp.sum()} abs_diff_sum={da} (boundary float ties allowed)")
ok &= da <= max(5, 0.001 * mg.anomalies_py.sum())
print("VERIFICATION", "PASS" if ok else "FAIL", f"({len(mg)} bins)")
