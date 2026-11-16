// common.h - shared data loader, bin structure and helpers for Phase 1 (sequential + OpenMP).
// Data layout: Structure-of-Arrays (SoA) so each pass streams only the columns it needs
// (better spatial locality / fewer cache lines than an Array-of-Structures; see cache test).
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <string>

constexpr int NZONES = 266;            // location ids 1..265 (index 0 unused)
constexpr int NHOURS = 24;
constexpr int NBINS  = NZONES * NHOURS; // one bin per (pickup zone, pickup hour) = 6384
constexpr double HARD_FPM = 100.0;      // hard anomaly: fare per mile > $100
constexpr double Z_THRESH = 3.0;        // statistical anomaly: fare/mile > mean + 3*sd of its bin
constexpr uint64_t MIN_TRIPS_Z = 30;    // need >= 30 trips in a bin for the z-score rule

struct Data {
    size_t n = 0;
    std::vector<uint16_t> bin;   // zone*24 + hour
    std::vector<float> dist, fare, tip, total;
};

// One cache line (64 B) per bin: avoids false sharing between neighbouring bins.
struct alignas(64) Bin {
    uint64_t trips = 0;
    double sum_dist = 0, sum_fare = 0, sum_total = 0;
    double sum_tippct = 0;        // sum of 100*tip/fare
    double sum_fpm = 0, sum_fpm2 = 0;   // sum and sum of squares of fare-per-mile
    uint64_t anomalies = 0;       // pass 2: z-score anomalies
};

struct Result {
    std::vector<Bin> bins = std::vector<Bin>(NBINS);
    uint64_t hard_anomalies = 0;  // global count of fare/mile > HARD_FPM
    double max_fpm = 0;           // global maximum fare per mile
    uint64_t total_trips = 0;
};

inline double now_sec() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}
inline double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t k = v.size();
    return k % 2 ? v[k / 2] : 0.5 * (v[k / 2 - 1] + v[k / 2]);
}

// ---- fast CSV loader (numeric columns only, 2-decimal numbers, '\n' line endings) -------------
// Columns: PULocationID,DOLocationID,pickup_hour,passenger_count,trip_distance,fare_amount,tip_amount,total_amount
static inline double parse_num(const char*& p) {
    double v = 0; bool neg = false;
    if (*p == '-') { neg = true; ++p; }
    while (*p >= '0' && *p <= '9') v = v * 10 + (*p++ - '0');
    if (*p == '.') {
        ++p; double f = 0.1;
        while (*p >= '0' && *p <= '9') { v += (*p++ - '0') * f; f *= 0.1; }
    }
    return neg ? -v : v;
}

inline bool load_csv(const char* path, Data& d) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return false; }
#ifdef _WIN32
    _fseeki64(f, 0, SEEK_END); long long fsz = _ftelli64(f); _fseeki64(f, 0, SEEK_SET);
#else
    fseeko(f, 0, SEEK_END); long long fsz = ftello(f); fseeko(f, 0, SEEK_SET);
#endif
    size_t est = (size_t)(fsz / 36) + 1024;
    d.bin.reserve(est); d.dist.reserve(est); d.fare.reserve(est); d.tip.reserve(est); d.total.reserve(est);
    const size_t CH = 1 << 26;
    std::vector<char> buf(CH + 1);
    size_t have = 0; bool header = true;
    while (true) {
        size_t r = fread(buf.data() + have, 1, CH - have, f);
        have += r;
        bool eof = (r == 0);
        size_t end = have;
        if (!eof) { while (end > 0 && buf[end - 1] != '\n') --end; } // process only complete lines
        size_t pos = 0;
        while (pos < end) {
            const char* p = buf.data() + pos;
            const char* nl = (const char*)memchr(p, '\n', end - pos);
            if (!nl) break;
            if (header) { header = false; pos = nl - buf.data() + 1; continue; }
            int zone = (int)parse_num(p); ++p;
            parse_num(p); ++p;                      // DOLocationID (unused)
            int hour = (int)parse_num(p); ++p;
            parse_num(p); ++p;                      // passenger_count (unused)
            float dist = (float)parse_num(p); ++p;
            float fare = (float)parse_num(p); ++p;
            float tip = (float)parse_num(p); ++p;
            float total = (float)parse_num(p);
            d.bin.push_back((uint16_t)(zone * NHOURS + hour));
            d.dist.push_back(dist); d.fare.push_back(fare); d.tip.push_back(tip); d.total.push_back(total);
            pos = nl - buf.data() + 1;
        }
        // move leftover partial line to front
        size_t left = have - pos;
        memmove(buf.data(), buf.data() + pos, left);
        have = left;
        if (eof) break;
    }
    fclose(f);
    d.n = d.bin.size();
    return true;
}

// ---- result comparison (counts exact, sums within relative tolerance: summation order differs) ----
inline bool close_rel(double a, double b, double tol = 1e-9) {
    return std::fabs(a - b) <= tol * std::max(1.0, std::max(std::fabs(a), std::fabs(b)));
}
inline bool compare_results(const Result& a, const Result& b, const char* label) {
    bool ok = a.total_trips == b.total_trips && a.hard_anomalies == b.hard_anomalies && close_rel(a.max_fpm, b.max_fpm);
    int bad = 0;
    for (int i = 0; i < NBINS; ++i) {
        const Bin &x = a.bins[i], &y = b.bins[i];
        bool e = x.trips == y.trips && x.anomalies == y.anomalies && close_rel(x.sum_dist, y.sum_dist) &&
                 close_rel(x.sum_fare, y.sum_fare) && close_rel(x.sum_total, y.sum_total) &&
                 close_rel(x.sum_tippct, y.sum_tippct) && close_rel(x.sum_fpm, y.sum_fpm) &&
                 close_rel(x.sum_fpm2, y.sum_fpm2);
        if (!e) ++bad;
    }
    ok = ok && bad == 0;
    printf("VERIFY,%s,%s,mismatched_bins=%d\n", label, ok ? "PASS" : "FAIL", bad);
    return ok;
}

inline void write_results(const Result& r, const char* path) {
    FILE* f = fopen(path, "w");
    if (!f) return;
    fprintf(f, "zone,hour,trips,avg_distance,total_fare,total_revenue,avg_tip_pct,avg_fare_per_mile,anomalies\n");
    for (int z = 0; z < NZONES; ++z)
        for (int h = 0; h < NHOURS; ++h) {
            const Bin& b = r.bins[z * NHOURS + h];
            if (!b.trips) continue;
            fprintf(f, "%d,%d,%llu,%.6f,%.2f,%.2f,%.6f,%.6f,%llu\n", z, h, (unsigned long long)b.trips,
                    b.sum_dist / b.trips, b.sum_fare, b.sum_total, b.sum_tippct / b.trips,
                    b.sum_fpm / b.trips, (unsigned long long)b.anomalies);
        }
    fclose(f);
}

inline void print_checksum(const Result& r) {
    double fare = 0, total = 0; uint64_t trips = 0, an = 0;
    for (const Bin& b : r.bins) { trips += b.trips; fare += b.sum_fare; total += b.sum_total; an += b.anomalies; }
    printf("CHECKSUM,trips=%llu,total_fare=%.2f,total_revenue=%.2f,z_anomalies=%llu,hard_anomalies=%llu,max_fpm=%.2f\n",
           (unsigned long long)trips, fare, total, (unsigned long long)an,
           (unsigned long long)r.hard_anomalies, r.max_fpm);
}

// threshold per bin for the z-score rule (serial, tiny: 6384 bins)
inline void compute_thresholds(const Result& r, std::vector<double>& thr) {
    thr.assign(NBINS, 1e300);
    for (int i = 0; i < NBINS; ++i) {
        const Bin& b = r.bins[i];
        if (b.trips < MIN_TRIPS_Z) continue;
        double m = b.sum_fpm / b.trips;
        double var = std::max(0.0, b.sum_fpm2 / b.trips - m * m);
        thr[i] = m + Z_THRESH * std::sqrt(var);
    }
}
