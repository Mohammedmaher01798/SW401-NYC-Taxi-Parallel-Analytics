# run_experiments.ps1 - compile (-O3 -fopenmp) and run all Phase 1 experiments on Windows.
# Usage (from repo root):  powershell -ExecutionPolicy Bypass -File plots\run_experiments.ps1
# Produces: results\*.log, plots\benchmark_summary.csv (via make_plots.py), plots\*.png
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
New-Item -ItemType Directory -Force -Path results, bin | Out-Null

# make sure g++ is on PATH (WinLibs installed through winget)
# NOTE: MinGW's linker breaks on install paths containing spaces, so the toolchain was copied to C:\mingw64.
if (Test-Path C:\mingw64\bin\g++.exe) { $env:Path = "C:\mingw64\bin;$env:Path" }
if (-not (Get-Command g++ -ErrorAction SilentlyContinue)) { throw "g++ not found (install WinLibs/MinGW-w64)" }
g++ --version | Select-Object -First 1

Write-Host "== Compile =="
g++ -O3 -fopenmp -std=c++17 src/sequential.cpp   -o bin/sequential.exe
g++ -O3 -fopenmp -std=c++17 src/parallel_omp.cpp -o bin/parallel_omp.exe

Write-Host "== Self-test (known-answer test) =="
& bin/sequential.exe --selftest | Tee-Object results\selftest.log

$REPS = 5      # >= 3 repetitions; median reported
$full = "data/clean_full.csv"; $s100 = "data/sample_100mb.csv"; $s500 = "data/sample_500mb.csv"

Write-Host "== Sequential baseline T_S (full, 500MB, 100MB samples) =="
& bin/sequential.exe $full $REPS results/seq_results_full.csv | Tee-Object results\seq_full.log
& bin/sequential.exe $s500 $REPS | Tee-Object results\seq_500mb.log
& bin/sequential.exe $s100 $REPS results/seq_results_100mb.csv | Tee-Object results\seq_100mb.log

Write-Host "== OpenMP scaling 1/2/4/8 threads (full) =="
& bin/parallel_omp.exe $full scaling $REPS | Tee-Object results\omp_scaling_full.log
& bin/parallel_omp.exe $s500 scaling $REPS | Tee-Object results\omp_scaling_500mb.log
& bin/parallel_omp.exe $s100 scaling $REPS | Tee-Object results\omp_scaling_100mb.log

Write-Host "== Scheduling comparison (8 threads, full) =="
& bin/parallel_omp.exe $full sched $REPS | Tee-Object results\omp_sched_full.log

Write-Host "== Synchronisation strategies (8 threads, 100MB sample) =="
& bin/parallel_omp.exe $s100 sync $REPS | Tee-Object results\omp_sync_100mb.log

Write-Host "== Cache / locality experiments (full) =="
& bin/parallel_omp.exe $full cache $REPS | Tee-Object results\omp_cache_full.log

Write-Host "== Plots =="
python plots/make_plots.py
