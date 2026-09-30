// =============================================================================
// PRACTICE 7 - C++ multithreading: data race, mutex, atomic, deadlock (Lecture 2)
//
// TASK:
//   Using std::thread, sum a vector of N = 1 << 24 integers (value = i % 10)
//   with T = std::thread::hardware_concurrency() threads, in four ways:
//     1. racySum      - every thread adds directly to a shared long long
//                       (shows the data race: result is usually wrong);
//     2. mutexSum     - each thread sums its chunk locally, then adds the
//                       partial result under a std::mutex (lock_guard);
//     3. atomicSum    - partial results added to a std::atomic<long long>;
//     4. sequentialSum- host reference.
//   Then show a deadlock-free money transfer between two accounts that two
//   threads call in OPPOSITE order (A->B and B->A), using std::scoped_lock.
//   Verify each sum and time each version.
//
// Build: g++ -std=c++17 -O2 -pthread -o threads 08_cpp_threads.cpp
// (nvcc also compiles it: nvcc -std=c++17 -o threads 08_cpp_threads.cpp)
// =============================================================================
#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <thread>
#include <vector>

const int N = 1 << 24;

// Chunk [begin, end) for thread t out of T
static void chunk(int t, int T, int n, int &begin, int &end)
{
    int size = (n + T - 1) / T;
    begin = t * size;
    end = begin + size < n ? begin + size : n;
}

long long racySum(const std::vector<int> &v, int T)
{
    long long total = 0;                        // shared, unprotected
    std::vector<std::thread> threads;
    for (int t = 0; t < T; ++t)
        threads.emplace_back([&, t] {
            int b, e; chunk(t, T, (int)v.size(), b, e);
            for (int i = b; i < e; ++i) total += v[i];   // DATA RACE: read-modify-write
        });
    for (auto &th : threads) th.join();
    return total;
}

long long mutexSum(const std::vector<int> &v, int T)
{
    long long total = 0;
    std::mutex m;
    std::vector<std::thread> threads;
    for (int t = 0; t < T; ++t)
        threads.emplace_back([&, t] {
            int b, e; chunk(t, T, (int)v.size(), b, e);
            long long local = 0;                // no sharing inside the hot loop
            for (int i = b; i < e; ++i) local += v[i];
            std::lock_guard<std::mutex> lock(m); // mutual exclusion, released at scope end
            total += local;
        });
    for (auto &th : threads) th.join();
    return total;
}

long long atomicSum(const std::vector<int> &v, int T)
{
    std::atomic<long long> total{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < T; ++t)
        threads.emplace_back([&, t] {
            int b, e; chunk(t, T, (int)v.size(), b, e);
            long long local = 0;
            for (int i = b; i < e; ++i) local += v[i];
            total += local;                     // atomic read-modify-write
        });
    for (auto &th : threads) th.join();
    return total.load();
}

long long sequentialSum(const std::vector<int> &v)
{
    long long s = 0;
    for (int x : v) s += x;
    return s;
}

// ---- Deadlock demo ---------------------------------------------------------
struct Account {
    std::mutex m;
    long long balance = 1000;
};

// WRONG (can deadlock): lock(from.m) then lock(to.m). Thread 1 does A->B,
// thread 2 does B->A: each holds one lock and waits forever for the other.
// RIGHT: std::scoped_lock locks both mutexes at once with a deadlock-avoidance
// algorithm (same idea as "always lock in the same order").
void transfer(Account &from, Account &to, long long amount)
{
    std::scoped_lock lock(from.m, to.m);
    from.balance -= amount;
    to.balance += amount;
}

template <typename F>
double timeMs(F f)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    f();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

int main()
{
    int T = (int)std::thread::hardware_concurrency();
    if (T == 0) T = 4;
    std::vector<int> v(N);
    for (int i = 0; i < N; ++i) v[i] = i % 10;

    long long ref = 0, racy = 0, mtx = 0, atm = 0;
    double tRef = timeMs([&] { ref = sequentialSum(v); });
    double tRacy = timeMs([&] { racy = racySum(v, T); });
    double tMtx = timeMs([&] { mtx = mutexSum(v, T); });
    double tAtm = timeMs([&] { atm = atomicSum(v, T); });

    printf("threads: %d\n", T);
    printf("sequential %lld  (%.2f ms)\n", ref, tRef);
    printf("racy       %lld  (%.2f ms) %s\n", racy, tRacy, racy == ref ? "correct (by luck)" : "WRONG - data race");
    printf("mutex      %lld  (%.2f ms) %s\n", mtx, tMtx, mtx == ref ? "CORRECT" : "WRONG");
    printf("atomic     %lld  (%.2f ms) %s\n", atm, tAtm, atm == ref ? "CORRECT" : "WRONG");

    Account A, B;
    std::thread t1([&] { for (int i = 0; i < 100000; ++i) transfer(A, B, 1); });
    std::thread t2([&] { for (int i = 0; i < 100000; ++i) transfer(B, A, 1); });
    t1.join();
    t2.join();
    printf("after opposite transfers: A = %lld, B = %lld (total %lld, no deadlock)\n",
           A.balance, B.balance, A.balance + B.balance);
    return (mtx == ref && atm == ref && A.balance + B.balance == 2000) ? 0 : 1;
}
