// sequential.cpp - verified sequential baseline (T_S) for per-(pickup zone, hour) aggregation
// + anomaly detection on NYC TLC Yellow Taxi data.
//
// Pass 1: per (zone,hour) bin: trips, sum distance, sum fare, sum total (revenue), sum tip%, sum/sumsq fare-per-mile
//         plus global: max fare-per-mile and count of "hard" anomalies (fare/mile > $100).
// Pass 2: statistical anomalies: trips whose fare-per-mile > mean + 3*sd of their own bin (bins with >= 30 trips).
//
// Build : g++ -O3 -fopenmp src/sequential.cpp -o sequential
// Run   : sequential <clean.csv> [reps=5] [results.csv]      |  sequential --selftest
#include "common.h"

// Serial kernel. This is the reference implementation used for T_S and to verify the parallel code.
void aggregate_serial(const Data& d, Result& r, double* t_pass1 = nullptr, double* t_pass2 = nullptr) {
    double t0 = now_sec();
    r = Result();
    const size_t n = d.n;
    const uint16_t* B = d.bin.data();
    const float *D = d.dist.data(), *F = d.fare.data(), *TP = d.tip.data(), *TO = d.total.data();
    Bin* bins = r.bins.data();
    uint64_t hard = 0; double maxf = 0;
    for (size_t i = 0; i < n; ++i) {
        double dist = D[i], fare = F[i];
        double fpm = fare / dist;
        Bin& b = bins[B[i]];
        b.trips += 1;
        b.sum_dist += dist;
        b.sum_fare += fare;
        b.sum_total += TO[i];
        b.sum_tippct += 100.0 * TP[i] / fare;
        b.sum_fpm += fpm;
        b.sum_fpm2 += fpm * fpm;
        if (fpm > HARD_FPM) ++hard;
        if (fpm > maxf) maxf = fpm;
    }
    r.hard_anomalies = hard; r.max_fpm = maxf; r.total_trips = d.n;
    double t1 = now_sec();
    std::vector<double> thr;
    compute_thresholds(r, thr);
    const double* TH = thr.data();
    for (size_t i = 0; i < n; ++i) {
        double fpm = (double)F[i] / (double)D[i];
        if (fpm > TH[B[i]]) bins[B[i]].anomalies += 1;
    }
    double t2 = now_sec();
    if (t_pass1) *t_pass1 = t1 - t0;
    if (t_pass2) *t_pass2 = t2 - t1;
}

#ifndef SEQ_NO_MAIN
// Known-answer test with hand-computed values.
static int selftest() {
    Data d;
    auto add = [&](int zone, int hour, float dist, float fare, float tip, float total) {
        d.bin.push_back((uint16_t)(zone * NHOURS + hour)); d.dist.push_back(dist);
        d.fare.push_back(fare); d.tip.push_back(tip); d.total.push_back(total);
    };
    add(1, 0, 2, 10, 2, 15);     // bin(1,0): 2 trips, avg dist 3, fare 30, revenue 40, avg tip% (20+0)/2 = 10
    add(1, 0, 4, 20, 0, 25);
    add(2, 5, 0.5f, 100, 5, 110); // fpm 200 -> exactly 1 hard anomaly
    for (int k = 0; k < 100; ++k) add(3, 7, 2, 10, 1, 12);  // fpm 5, 100 trips
    add(3, 7, 1, 100, 0, 100);                               // fpm 100 -> z-score anomaly (not > 100 so not hard)
    d.n = d.bin.size();
    Result r; aggregate_serial(d, r);
    int fails = 0;
    auto chk = [&](bool c, const char* what) { if (!c) { ++fails; printf("SELFTEST FAIL: %s\n", what); } };
    const Bin& a = r.bins[1 * NHOURS + 0];
    chk(a.trips == 2, "trips(1,0)=2");
    chk(close_rel(a.sum_dist / a.trips, 3.0, 1e-9), "avg dist(1,0)=3");
    chk(close_rel(a.sum_fare, 30.0), "total fare(1,0)=30");
    chk(close_rel(a.sum_total, 40.0), "revenue(1,0)=40");
    chk(close_rel(a.sum_tippct / a.trips, 10.0), "avg tip% (1,0)=10");
    chk(a.anomalies == 0, "no anomalies (1,0)");
    const Bin& c = r.bins[3 * NHOURS + 7];
    chk(c.trips == 101, "trips(3,7)=101");
    chk(c.anomalies == 1, "exactly 1 z-score anomaly in (3,7)");
    chk(r.hard_anomalies == 1, "exactly 1 hard anomaly");
    chk(close_rel(r.max_fpm, 200.0), "max fare per mile = 200");
    printf("SELFTEST,%s\n", fails ? "FAIL" : "PASS");
    return fails;
}

int main(int argc, char** argv) {
    if (argc >= 2 && !strcmp(argv[1], "--selftest")) return selftest();
    if (argc < 2) { fprintf(stderr, "usage: %s <clean.csv> [reps=5] [results.csv]\n", argv[0]); return 1; }
    int reps = argc > 2 ? atoi(argv[2]) : 5;
    Data d;
    double tl = now_sec();
    if (!load_csv(argv[1], d)) return 1;
    printf("DATASET,%s,rows=%zu\n", argv[1], d.n);
    printf("LOAD_SECONDS,%.4f\n", now_sec() - tl);
    std::vector<double> ts, p1, p2;
    Result r;
    aggregate_serial(d, r); // warm-up pass (matching parallel_omp.cpp benchmark protocol)
    for (int k = 0; k < reps; ++k) {
        double a, b; double t0 = now_sec();
        aggregate_serial(d, r, &a, &b);
        double t = now_sec() - t0;
        ts.push_back(t); p1.push_back(a); p2.push_back(b);
        printf("SEQ_TIME,%d,%.6f,pass1=%.6f,pass2=%.6f\n", k + 1, t, a, b);
    }
    printf("SEQ_MEDIAN,%.6f,pass1=%.6f,pass2=%.6f\n", median(ts), median(p1), median(p2));
    print_checksum(r);
    uint64_t sum = 0; for (auto& b : r.bins) sum += b.trips;
    printf("INVARIANT,sum_of_bin_trips==rows,%s\n", sum == d.n ? "PASS" : "FAIL");
    if (argc > 3) write_results(r, argv[3]);
    return 0;
}
#endif
