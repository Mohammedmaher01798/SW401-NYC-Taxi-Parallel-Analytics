// parallel_omp.cpp - OpenMP version of the zone/hour aggregation + anomaly detection, plus
// scheduling comparison, synchronisation-strategy comparison and cache/locality experiments.
//
// Build : g++ -O3 -fopenmp src/parallel_omp.cpp -o parallel_omp
// Run   : parallel_omp <clean.csv> <mode> [reps=5]
//   modes: scaling | sched | sync | cache | all
// Output lines are machine readable (SCALE,... SCHED,... SYNC,... CACHE,... VERIFY,...).
#define SEQ_NO_MAIN
#include "sequential.cpp"      // re-use aggregate_serial() as the correctness reference
#include <omp.h>

enum Sync { LOCAL, ATOMIC, CRITICAL };

// Parallel kernel.
//  * LOCAL    : each thread aggregates into a private array of Bins (no sharing, no locks); the private arrays
//               are merged in parallel over bins. Scalars (hard anomalies, max fpm) use reduction clauses.
//  * ATOMIC   : all threads update one shared array using '#pragma omp atomic' per field (contention demo).
//  * CRITICAL : one shared array protected by '#pragma omp critical' (worst case demo).
// Loop scheduling is selected at run time via omp_set_schedule() + schedule(runtime).
void aggregate_omp(const Data& d, Result& r, int T, Sync mode, double* t1p = nullptr, double* t2p = nullptr) {
    double t0 = now_sec();
    r = Result();
    const long n = (long)d.n;
    Bin* shared = r.bins.data();
    std::vector<std::vector<Bin>> local(T);
    uint64_t hard = 0; double maxf = 0;
    const uint16_t* B = d.bin.data(); const float *D = d.dist.data(), *F = d.fare.data(),
                    *TP = d.tip.data(), *TO = d.total.data();

    #pragma omp parallel num_threads(T) reduction(+:hard) reduction(max:maxf)
    {
        std::vector<Bin>& loc = local[omp_get_thread_num()];
        if (mode == LOCAL) loc.assign(NBINS, Bin());   // first-touch by owning thread
        Bin* lb = loc.data();
        #pragma omp for schedule(runtime)
        for (long i = 0; i < n; ++i) {
            double dist = D[i], fare = F[i];
            double fpm = fare / dist;
            double tp = 100.0 * TP[i] / fare, tot = TO[i];
            if (fpm > HARD_FPM) ++hard;
            if (fpm > maxf) maxf = fpm;
            if (mode == LOCAL) {
                Bin& b = lb[B[i]];
                b.trips += 1; b.sum_dist += dist; b.sum_fare += fare; b.sum_total += tot;
                b.sum_tippct += tp; b.sum_fpm += fpm; b.sum_fpm2 += fpm * fpm;
            } else if (mode == ATOMIC) {
                Bin& b = shared[B[i]];
                #pragma omp atomic
                b.trips += 1;
                #pragma omp atomic
                b.sum_dist += dist;
                #pragma omp atomic
                b.sum_fare += fare;
                #pragma omp atomic
                b.sum_total += tot;
                #pragma omp atomic
                b.sum_tippct += tp;
                #pragma omp atomic
                b.sum_fpm += fpm;
                #pragma omp atomic
                b.sum_fpm2 += fpm * fpm;
            } else {
                #pragma omp critical(bins_update)
                {
                    Bin& b = shared[B[i]];
                    b.trips += 1; b.sum_dist += dist; b.sum_fare += fare; b.sum_total += tot;
                    b.sum_tippct += tp; b.sum_fpm += fpm; b.sum_fpm2 += fpm * fpm;
                }
            }
        }
    }
    if (mode == LOCAL) {   // parallel merge over bins: each bin owned by exactly one thread
        #pragma omp parallel for num_threads(T) schedule(static)
        for (int k = 0; k < NBINS; ++k) {
            Bin& o = shared[k];
            for (int t = 0; t < T; ++t) {
                const Bin& s = local[t][k];
                o.trips += s.trips; o.sum_dist += s.sum_dist; o.sum_fare += s.sum_fare;
                o.sum_total += s.sum_total; o.sum_tippct += s.sum_tippct;
                o.sum_fpm += s.sum_fpm; o.sum_fpm2 += s.sum_fpm2;
            }
        }
    }
    r.hard_anomalies = hard; r.max_fpm = maxf; r.total_trips = d.n;
    double t1 = now_sec();

    // ---- pass 2: z-score anomalies, same pattern (thread-local counters + merge) ----
    std::vector<double> thr; compute_thresholds(r, thr);
    const double* TH = thr.data();
    std::vector<std::vector<uint64_t>> lan(T);
    #pragma omp parallel num_threads(T)
    {
        std::vector<uint64_t>& la = lan[omp_get_thread_num()];
        la.assign(NBINS, 0);
        #pragma omp for schedule(runtime)
        for (long i = 0; i < n; ++i) {
            double fpm = (double)F[i] / (double)D[i];
            if (fpm > TH[B[i]]) la[B[i]] += 1;
        }
    }
    #pragma omp parallel for num_threads(T) schedule(static)
    for (int k = 0; k < NBINS; ++k)
        for (int t = 0; t < T; ++t) shared[k].anomalies += lan[t][k];
    double t2 = now_sec();
    if (t1p) *t1p = t1 - t0;
    if (t2p) *t2p = t2 - t1;
}

static const char* kind_name(omp_sched_t k) {
    return k == omp_sched_static ? "static" : k == omp_sched_dynamic ? "dynamic" : "guided";
}

static double timed(const Data& d, int T, Sync m, omp_sched_t kind, int chunk, Result& r) {
    omp_set_schedule(kind, chunk);
    double t0 = now_sec();
    aggregate_omp(d, r, T, m);
    return now_sec() - t0;
}

// ---------------------------------------------------------------------------------------------------
static void mode_scaling(const Data& d, const Result& ref, int reps) {
    for (int T : {1, 2, 4, 8}) {
        Result r;
        timed(d, T, LOCAL, omp_sched_static, 0, r);              // warm-up (not recorded)
        compare_results(ref, r, ("scaling_T" + std::to_string(T)).c_str());
        for (int k = 0; k < reps; ++k) {
            omp_set_schedule(omp_sched_static, 0);
            double a, b, t0 = now_sec();
            aggregate_omp(d, r, T, LOCAL, &a, &b);
            printf("SCALE,%d,%d,%.6f,%.6f,%.6f\n", T, k + 1, now_sec() - t0, a, b);
        }
        fflush(stdout);
    }
}

static void mode_sched(const Data& d, const Result& ref, int reps, int T) {
    struct Cfg { omp_sched_t k; int c; };
    std::vector<Cfg> cfgs;
    for (omp_sched_t k : {omp_sched_static, omp_sched_dynamic, omp_sched_guided})
        for (int c : (k == omp_sched_static ? std::vector<int>{0, 64, 4096, 262144}
                                            : std::vector<int>{64, 4096, 262144}))
            cfgs.push_back({k, c});
    for (auto& cf : cfgs) {
        Result r;
        timed(d, T, LOCAL, cf.k, cf.c, r);
        compare_results(ref, r, (std::string(kind_name(cf.k)) + "_" + std::to_string(cf.c)).c_str());
        for (int k = 0; k < reps; ++k)
            printf("SCHED,%s,%d,%d,%d,%.6f\n", kind_name(cf.k), cf.c, T, k + 1, timed(d, T, LOCAL, cf.k, cf.c, r));
        fflush(stdout);
    }
}

static void mode_sync(const Data& d, const Result& ref, int reps, int T) {
    const char* nm[] = {"thread_local_arrays", "atomic_shared", "critical_shared"};
    for (Sync m : {LOCAL, ATOMIC, CRITICAL}) {
        Result r; timed(d, T, m, omp_sched_static, 0, r);
        compare_results(ref, r, nm[m]);
        for (int k = 0; k < reps; ++k)
            printf("SYNC,%s,%d,%d,%.6f\n", nm[m], T, k + 1, timed(d, T, m, omp_sched_static, 0, r));
        fflush(stdout);
    }
}

// ---- cache / memory-hierarchy experiments ----------------------------------------------------------------
volatile double g_sink; volatile long g_lsink;

static void mode_cache(const Data& d, int reps, int T) {
    const long n = (long)d.n;
    const float* F = d.fare.data();
    // (A) Strided vs contiguous traversal. Every stride visits ALL n elements exactly once (s interleaved passes),
    //     so the instruction count is identical; only the access pattern (cache-line reuse, prefetching) differs.
    for (int s : {1, 2, 4, 8, 16, 32, 64, 1024}) {
        std::vector<double> ts, tp;
        for (int k = 0; k < reps; ++k) {
            double t0 = now_sec(); double sum = 0;
            for (int off = 0; off < s; ++off) for (long i = off; i < n; i += s) sum += F[i];
            ts.push_back(now_sec() - t0); g_sink = sum;
            t0 = now_sec(); double psum = 0;
            for (int off = 0; off < s; ++off) {
                #pragma omp parallel for num_threads(T) reduction(+:psum) schedule(static)
                for (long i = off; i < n; i += s) psum += F[i];
            }
            tp.push_back(now_sec() - t0); g_sink = psum;
        }
        printf("CACHE,stride,%d,serial,%.6f\n", s, median(ts));
        printf("CACHE,stride,%d,omp%d,%.6f\n", s, T, median(tp));
    }
    // (B) Array-of-Structures (24 B/row) vs Structure-of-Arrays (6 B/row used: bin + fare)
    //     for both serial and 8-thread OpenMP (shows memory-bandwidth reduction from SoA).
    struct Row { uint16_t bin; float dist, fare, tip, total; };       // 24 bytes with padding
    std::vector<Row> aos(n);
    for (long i = 0; i < n; ++i) aos[i] = {d.bin[i], d.dist[i], d.fare[i], d.tip[i], d.total[i]};
    {
        std::vector<double> ta, tsoa, ta8, tsoa8;
        std::vector<double> acc(NBINS);
        std::vector<std::vector<double>> lacc(T, std::vector<double>(NBINS));
        for (int k = 0; k < reps; ++k) {
            std::fill(acc.begin(), acc.end(), 0.0);
            double t0 = now_sec();
            for (long i = 0; i < n; ++i) acc[aos[i].bin] += aos[i].fare;
            ta.push_back(now_sec() - t0); g_sink = acc[5];
            std::fill(acc.begin(), acc.end(), 0.0);
            t0 = now_sec();
            for (long i = 0; i < n; ++i) acc[d.bin[i]] += d.fare[i];
            tsoa.push_back(now_sec() - t0); g_sink = acc[5];

            t0 = now_sec();
            #pragma omp parallel num_threads(T)
            {
                auto& la = lacc[omp_get_thread_num()];
                std::fill(la.begin(), la.end(), 0.0);
                #pragma omp for schedule(static)
                for (long i = 0; i < n; ++i) la[aos[i].bin] += aos[i].fare;
            }
            ta8.push_back(now_sec() - t0); g_sink = lacc[0][5];

            t0 = now_sec();
            #pragma omp parallel num_threads(T)
            {
                auto& la = lacc[omp_get_thread_num()];
                std::fill(la.begin(), la.end(), 0.0);
                #pragma omp for schedule(static)
                for (long i = 0; i < n; ++i) la[d.bin[i]] += d.fare[i];
            }
            tsoa8.push_back(now_sec() - t0); g_sink = lacc[0][5];
        }
        printf("CACHE,layout,AoS_24B,serial,%.6f\n", median(ta));
        printf("CACHE,layout,SoA_6B,serial,%.6f\n", median(tsoa));
        printf("CACHE,layout,AoS_24B,omp%d,%.6f\n", T, median(ta8));
        printf("CACHE,layout,SoA_6B,omp%d,%.6f\n", T, median(tsoa8));
    }
    aos.clear(); aos.shrink_to_fit();
    // (C) False sharing: per-thread counters (volatile forces real memory updates every iteration).
    {
        std::vector<long> adj(T * 16, 0);      // adjacent 8-byte counters: several threads share a cache line
        std::vector<long> pad(T * 16, 0);      // one counter per 128 bytes
        long iters = std::min<long>(n, 40000000);
        std::vector<double> tadj, tpad, tloc;
        for (int k = 0; k < reps; ++k) {
            double t0 = now_sec();
            #pragma omp parallel num_threads(T)
            { volatile long* c = &adj[omp_get_thread_num()];
              #pragma omp for schedule(static)
              for (long i = 0; i < iters; ++i) if (F[i % n] > 10.f) (*c)++; }
            tadj.push_back(now_sec() - t0);
            t0 = now_sec();
            #pragma omp parallel num_threads(T)
            { volatile long* c = &pad[omp_get_thread_num() * 16];
              #pragma omp for schedule(static)
              for (long i = 0; i < iters; ++i) if (F[i % n] > 10.f) (*c)++; }
            tpad.push_back(now_sec() - t0);
            t0 = now_sec(); long tot = 0;
            #pragma omp parallel for num_threads(T) schedule(static) reduction(+:tot)
            for (long i = 0; i < iters; ++i) if (F[i % n] > 10.f) tot++;
            tloc.push_back(now_sec() - t0); g_lsink = tot;
        }
        printf("CACHE,falsesharing,adjacent_counters,omp%d,%.6f\n", T, median(tadj));
        printf("CACHE,falsesharing,padded_counters,omp%d,%.6f\n", T, median(tpad));
        printf("CACHE,falsesharing,reduction_clause,omp%d,%.6f\n", T, median(tloc));
    }
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <clean.csv> <scaling|sched|sync|cache|all> [reps=5]\n", argv[0]); return 1; }
    std::string mode = argv[2];
    int reps = argc > 3 ? atoi(argv[3]) : 5;
    Data d;
    double tl = now_sec();
    if (!load_csv(argv[1], d)) return 1;
    printf("DATASET,%s,rows=%zu,load_seconds=%.3f,max_threads=%d\n", argv[1], d.n, now_sec() - tl, omp_get_max_threads());
    Result ref; aggregate_serial(d, ref);                      // correctness reference
    print_checksum(ref);
    if (mode == "scaling" || mode == "all") mode_scaling(d, ref, reps);
    if (mode == "sched" || mode == "all") mode_sched(d, ref, reps, 8);
    if (mode == "sync" || mode == "all") mode_sync(d, ref, reps, 8);
    if (mode == "cache" || mode == "all") mode_cache(d, reps, 8);
    return 0;
}
