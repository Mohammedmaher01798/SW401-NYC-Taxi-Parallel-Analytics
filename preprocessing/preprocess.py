"""Phase 1 preprocessing: download 8 NYC TLC Yellow Taxi months, clean, export CSV + samples.

Usage:  python preprocessing/preprocess.py
Outputs (in ./data, git-ignored except small notes):
  raw/yellow_tripdata_2023-MM.parquet   (downloaded)
  clean_full.csv, sample_100mb.csv, sample_500mb.csv
  preprocessing_summary.txt             (before/after sizes and record counts)
"""
import os, sys, time, urllib.request
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RAW = os.path.join(DATA, "raw")
os.makedirs(RAW, exist_ok=True)
YEAR, MONTHS = 2023, range(1, 9)
URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{y}-{m:02d}.parquet"
FULL = os.path.join(DATA, "clean_full.csv")
S100 = os.path.join(DATA, "sample_100mb.csv")
S500 = os.path.join(DATA, "sample_500mb.csv")
SUMMARY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preprocessing_summary.txt")
# Output columns (all numeric): PULocationID, DOLocationID, pickup_hour, passenger_count,
# trip_distance, fare_amount, tip_amount, total_amount
COLS = ["PULocationID", "DOLocationID", "pickup_hour", "passenger_count",
        "trip_distance", "fare_amount", "tip_amount", "total_amount"]
log_lines = []
def log(s=""):
    print(s, flush=True); log_lines.append(s)

# ---- 1. download -------------------------------------------------------------------------
files = []
for m in MONTHS:
    p = os.path.join(RAW, f"yellow_tripdata_{YEAR}-{m:02d}.parquet")
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        u = URL.format(y=YEAR, m=m)
        log(f"Downloading {u}")
        urllib.request.urlretrieve(u, p)
    files.append((m, p))

# ---- 2. BEFORE stats ------------------------------------------------------------------------
before_rows, before_bytes, per_file = 0, 0, []
for m, p in files:
    n = pq.ParquetFile(p).metadata.num_rows
    b = os.path.getsize(p)
    per_file.append((os.path.basename(p), n, b))
    before_rows += n; before_bytes += b

# ---- 3. clean + export -------------------------------------------------------------------------
t0 = time.time()
after_rows = 0
drops = dict(null=0, nonpositive=0, bad_time=0, bad_zone=0, bad_dist=0)
if os.path.exists(FULL): os.remove(FULL)
for i, (m, p) in enumerate(files):
    df = pd.read_parquet(p, columns=["tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID",
        "DOLocationID", "passenger_count", "trip_distance", "fare_amount", "tip_amount", "total_amount"])
    n0 = len(df)
    df = df.dropna(); drops["null"] += n0 - len(df); n1 = len(df)
    df = df[(df.trip_distance > 0) & (df.fare_amount > 0) & (df.total_amount > 0) &
            (df.tip_amount >= 0) & (df.passenger_count > 0)]
    drops["nonpositive"] += n1 - len(df); n2 = len(df)
    # timestamps: keep only pickups inside the file's own month and dropoff after pickup
    pu = pd.to_datetime(df.tpep_pickup_datetime); do = pd.to_datetime(df.tpep_dropoff_datetime)
    ok = (pu.dt.year == YEAR) & (pu.dt.month == m) & (do > pu)
    df = df[ok]; pu = pu[ok]; drops["bad_time"] += n2 - len(df); n3 = len(df)
    df = df[(df.PULocationID.between(1, 265)) & (df.DOLocationID.between(1, 265))]
    drops["bad_zone"] += n3 - len(df); n4 = len(df)
    df = df[df.trip_distance < 1000]    # physically impossible distances (sensor errors)
    drops["bad_dist"] += n4 - len(df)
    out = pd.DataFrame({
        "PULocationID": df.PULocationID.astype("int32"), "DOLocationID": df.DOLocationID.astype("int32"),
        "pickup_hour": pu.loc[df.index].dt.hour.astype("int32"),
        "passenger_count": df.passenger_count.astype("int32"),
        "trip_distance": df.trip_distance.round(2), "fare_amount": df.fare_amount.round(2),
        "tip_amount": df.tip_amount.round(2), "total_amount": df.total_amount.round(2)})[COLS]
    out.to_csv(FULL, mode="a", header=(i == 0), index=False, lineterminator="\n", float_format="%.2f")
    after_rows += len(out)
    log(f"  month {m:02d}: {n0:,} -> {len(out):,} rows")
full_bytes = os.path.getsize(FULL)
log(f"Cleaned CSV written in {time.time()-t0:.1f}s")

# ---- 4. samples (nested Bernoulli sampling, fixed seed => 100MB subset of 500MB subset) ----------
rng = np.random.default_rng(401)
p500 = 500e6 / full_bytes; p100 = 100e6 / full_bytes
n100 = n500 = 0
for f in (S100, S500):
    if os.path.exists(f): os.remove(f)
first = True
for chunk in pd.read_csv(FULL, chunksize=2_000_000):
    u = rng.random(len(chunk))
    c5 = chunk[u < p500]; c1 = chunk[u < p100]
    c5.to_csv(S500, mode="a", header=first, index=False, lineterminator="\n", float_format="%.2f")
    c1.to_csv(S100, mode="a", header=first, index=False, lineterminator="\n", float_format="%.2f")
    first = False; n100 += len(c1); n500 += len(c5)

# ---- 5. summary ------------------------------------------------------------------------------------
log_lines.clear()
log("=== PREPROCESSING SUMMARY (NYC TLC Yellow Taxi, Jan-Aug 2023) ===")
log("BEFORE (raw Parquet, 8 monthly files):")
for name, n, b in per_file:
    log(f"  {name:36s} {n:>12,} records {b/1e6:>9.1f} MB")
log(f"  TOTAL raw: {before_rows:,} records, {before_bytes:,} bytes ({before_bytes/1e6:.1f} MB Parquet, compressed)")
log("Rows removed by cleaning step:")
log(f"  null values in used columns        : {drops['null']:,}")
log(f"  distance/fare/total<=0, tip<0, pax<=0 : {drops['nonpositive']:,}")
log(f"  pickup outside its month or dropoff<=pickup : {drops['bad_time']:,}")
log(f"  zone id outside 1..265             : {drops['bad_zone']:,}")
log(f"  trip_distance >= 1000 miles        : {drops['bad_dist']:,}")
log("AFTER:")
log(f"  clean_full.csv    : {after_rows:,} records, {full_bytes:,} bytes ({full_bytes/1e9:.3f} GB)")
log(f"  sample_100mb.csv  : {n100:,} records, {os.path.getsize(S100):,} bytes ({os.path.getsize(S100)/1e6:.1f} MB)")
log(f"  sample_500mb.csv  : {n500:,} records, {os.path.getsize(S500):,} bytes ({os.path.getsize(S500)/1e6:.1f} MB)")
log(f"  retained: {100*after_rows/before_rows:.2f}% of raw records")
log("  Columns: " + ",".join(COLS))
with open(SUMMARY, "w") as fh: fh.write("\n".join(log_lines) + "\n")
