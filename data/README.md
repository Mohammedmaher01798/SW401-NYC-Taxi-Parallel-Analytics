# Dataset Directory (`data/`)

Large raw Parquet files and cleaned CSV datasets are excluded from Git via `.gitignore` to keep the repository lightweight.

## Reproducing the Dataset Locally

Run the preprocessing pipeline from the repository root:

```bash
pip install pandas pyarrow numpy matplotlib
python preprocessing/preprocess.py
```

This automatically downloads the 8 NYC TLC Yellow Taxi Parquet files (January–August 2023) from:
`https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-{01..08}.parquet`

into `data/raw/` and generates:
- `data/clean_full.csv` — `23,592,248` records (`809,535,908` bytes / `0.810 GB`)
- `data/sample_500mb.csv` — `14,570,340` records (`499,963,267` bytes / `500.0 MB`)
- `data/sample_100mb.csv` — `2,914,997` records (`100,023,489` bytes / `100.0 MB`)

Full before/after statistics and cleaning rules are logged in [`preprocessing/preprocessing_summary.txt`](../preprocessing/preprocessing_summary.txt).
