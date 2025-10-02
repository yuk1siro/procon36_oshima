#include <bits/stdc++.h>  // ★ 追加: std::iota

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_CUDA
#include <cuda_runtime.h>
#define CUDA_CHECK(call) do { \
    cudaError_t __e = (call); \
    if (__e != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(__e)); \
        std::exit(1); \
    } \
} while(0)
#endif

#include <nlohmann/json.hpp>
using json = nlohmann::json;
using namespace std;

#if defined(_MSC_VER)
#define FINL __forceinline
#else
#define FINL __attribute__((always_inline)) inline
#endif

// ===== Consts =====
static const int MAXV = 300; // N<=24 なら N*N/2=288 以下

// ===== Field =====
struct Field {
    vector<int16_t> cells;
    int size;
    Field(int s = 0) : cells(s* s), size(s) {}
    FINL int16_t& at(int y, int x) { return cells[y * size + x]; }
    FINL int16_t  at(int y, int x) const { return cells[y * size + x]; }
};

// ===== Rotation map (16bit index) =====
struct RotMap {
    int x = 0, y = 0, s = 0;
    vector<uint16_t> src, dst;   // src[t] -> dst[t]
    vector<uint16_t> dst2src;    // (destオフセット) -> 元index（回転後の値参照をO(1)化）
};

// ===== Zobrist =====
struct Sig {
    uint64_t h1 = 0, h2 = 0;
    bool operator==(const Sig& o) const noexcept { return h1 == o.h1 && h2 == o.h2; }
};
struct SigHash {
    size_t operator()(const Sig& s) const noexcept {
        uint64_t x = s.h1 ^ (s.h2 + 0x9e3779b97f4a7c15ULL + (s.h1 << 6) + (s.h1 >> 2));
        return (size_t)x;
    }
};

// ===== visited set =====
#if __has_include(<absl/container/flat_hash_set.h>)
#include <absl/container/flat_hash_set.h>
template <class K, class H> using FastHashSet = absl::flat_hash_set<K, H>;
#elif __has_include("ankerl/unordered_dense.h")
#include "ankerl/unordered_dense.h"
template <class K, class H> using FastHashSet = ankerl::unordered_dense::set<K, H>;
#else
#include <unordered_set>
template <class K, class H> using FastHashSet = std::unordered_set<K, H>;
#endif

class App {
public:
    App(const string& jsonPath);
    void run();

    // Beam ノード
    struct Node_Beam {
        Field state;
        int g = 0, h = 0;
        int rx = 0, ry = 0, rs = 0;
        Sig sig{};
        shared_ptr<Node_Beam> parent;
        FINL int f() const { return g + h; }
        Node_Beam(int size) : state(size) {}
    };

    // ★ 修正: Cand を public に
    struct Cand {
        int f, h, g;
        int rx, ry, rs;
        shared_ptr<App::Node_Beam> parent;
        const RotMap* map;
    };

    // 通信系は public static として公開（main から呼べるように）
    static inline void cmd(const char cmd[]) {
        // system(cmd); // 必要なら有効化
    }

private:
    // ---- Heuristic Context（差分・1パス、完全版＋下界）----
    struct HeuCtx {
        int N;
        const Field* st;

        vector<uint8_t> cnt;
        vector<array<short, 2>> xs, ys;
        int curCost[MAXV];
        const vector<int>* rowBase;

        mutable vector<uint32_t> seenStamp;
        mutable uint32_t epoch = 1;

        int base = 0;

        // ★「非隣接ペア」を構成するセルのフラグ（小園・近傍バイアス用）
        vector<uint8_t> nonAdjMask;

        HeuCtx(const Field& f, const vector<int>& rowBase_)
            : N(f.size), st(&f), cnt(MAXV, 0), xs(MAXV), ys(MAXV), rowBase(&rowBase_) {

            for (int y = 0; y < N; ++y) {
                int baseIdx = (*rowBase)[y];
                for (int x = 0; x < N; ++x) {
                    int v = (int)f.cells[baseIdx + x];
                    if ((unsigned)v < (unsigned)MAXV) {
                        uint8_t c = cnt[v]++; if (c < 2) { xs[v][c] = (short)x; ys[v][c] = (short)y; }
                    }
                }
            }
            base = 0;
            for (int v = 0; v < MAXV; ++v) {
                if (cnt[v] == 2) {
                    int c = pairCostCurrent_cached(v);
                    curCost[v] = c; base += c;
                }
                else curCost[v] = 0;
            }

            // 非隣接ペアのセルをマーキング
            nonAdjMask.assign(N * N, 0);
            for (int v = 0; v < MAXV; ++v) if (cnt[v] == 2) {
                int x1 = xs[v][0], y1 = ys[v][0], x2 = xs[v][1], y2 = ys[v][1];
                if (std::abs(x1 - x2) + std::abs(y1 - y2) > 1) {
                    nonAdjMask[(*rowBase)[y1] + x1] = 1;
                    nonAdjMask[(*rowBase)[y2] + x2] = 1;
                }
            }

            seenStamp.assign(MAXV, 0);
        }

        FINL void begin_eval() const {
            if (++epoch == 0) { std::fill(seenStamp.begin(), seenStamp.end(), 0); epoch = 1; }
        }
        static FINL bool inside(const RotMap& m, int x, int y) {
            return (unsigned)(x - m.x) < (unsigned)m.s && (unsigned)(y - m.y) < (unsigned)m.s;
        }
        FINL void dest_xy(int xs_, int ys_, const RotMap& m, int& xd, int& yd) const {
            if ((unsigned)(xs_ - m.x) < (unsigned)m.s && (unsigned)(ys_ - m.y) < (unsigned)m.s) {
                xd = m.x + (m.s - 1 - (ys_ - m.y));
                yd = m.y + (xs_ - m.x);
            }
            else { xd = xs_; yd = ys_; }
        }
        FINL int val_after_at(const RotMap& m, int xd, int yd) const {
            if ((unsigned)(xd - m.x) < (unsigned)m.s && (unsigned)(yd - m.y) < (unsigned)m.s) {
                int off = (yd - m.y) * m.s + (xd - m.x);
                return (int)st->cells[(int)m.dst2src[off]];
            }
            return (int)st->cells[(*rowBase)[yd] + xd];
        }

        // 現在寄与（完全版）
        FINL int pairCostCurrent_cached(int v) const {
            int x1 = xs[v][0], y1 = ys[v][0], x2 = xs[v][1], y2 = ys[v][1];
            int dx = std::abs(x1 - x2), dy = std::abs(y1 - y2); int cost = dx + dy;
            if (dx == 0 && dy == 1) {
                int i1 = (*rowBase)[y1] + x1, i2 = (*rowBase)[y2] + x2;
                if (x1 + 1 < N && x2 + 1 < N) {
                    int r1 = (int)st->cells[i1 + 1], r2 = (int)st->cells[i2 + 1];
                    if (r1 == r2) cost -= 10;
                }
                cost -= 20;
            }
            else if (dx == 1 && dy == 0) {
                if (y1 + 1 < N && y2 + 1 < N) {
                    int d1 = (int)st->cells[(*rowBase)[y1 + 1] + x1];
                    int d2 = (int)st->cells[(*rowBase)[y2 + 1] + x2];
                    if (d1 == d2) cost -= 10;
                }
                cost -= 20;
            }
            return cost;
        }

        // 回転後寄与（完全版）
        FINL int pairCostAfter_full(const RotMap& m, int v) const {
            int x1s = xs[v][0], y1s = ys[v][0], x2s = xs[v][1], y2s = ys[v][1];
            int x1d, y1d, x2d, y2d; dest_xy(x1s, y1s, m, x1d, y1d); dest_xy(x2s, y2s, m, x2d, y2d);
            int dx = std::abs(x1d - x2d), dy = std::abs(y1d - y2d); int md = dx + dy; int cost = md;
            if (md == 1) {
                if (x1d == x2d) { // 縦
                    if (x1d + 1 < N) {
                        int in1 = (unsigned)(x1d + 1 - m.x) < (unsigned)m.s && (unsigned)(y1d - m.y) < (unsigned)m.s;
                        int in2 = (unsigned)(x2d + 1 - m.x) < (unsigned)m.s && (unsigned)(y2d - m.y) < (unsigned)m.s;
                        int r1 = in1 ? (int)st->cells[(int)m.dst2src[(y1d - m.y) * m.s + ((x1d + 1) - m.x)]]
                            : (int)st->cells[(*rowBase)[y1d] + (x1d + 1)];
                        int r2 = in2 ? (int)st->cells[(int)m.dst2src[(y2d - m.y) * m.s + ((x2d + 1) - m.x)]]
                            : (int)st->cells[(*rowBase)[y2d] + (x2d + 1)];
                        if (r1 == r2) cost -= 10;
                    }
                    cost -= 20;
                }
                else {          // 横
                    if (y1d + 1 < N) {
                        int in1 = (unsigned)(x1d - m.x) < (unsigned)m.s && (unsigned)(y1d + 1 - m.y) < (unsigned)m.s;
                        int in2 = (unsigned)(x2d - m.x) < (unsigned)m.s && (unsigned)(y2d + 1 - m.y) < (unsigned)m.s;
                        int d1 = in1 ? (int)st->cells[(int)m.dst2src[((y1d + 1) - m.y) * m.s + (x1d - m.x)]]
                            : (int)st->cells[(*rowBase)[y1d + 1] + x1d];
                        int d2 = in2 ? (int)st->cells[(int)m.dst2src[((y2d + 1) - m.y) * m.s + (x2d - m.x)]]
                            : (int)st->cells[(*rowBase)[y2d + 1] + x2d];
                        if (d1 == d2) cost -= 10;
                    }
                    cost -= 20;
                }
            }
            return cost;
        }

        // 回転後寄与の安全な下界（LB）
        FINL int pairCostAfter_LB(const RotMap& m, int v) const {
            int x1s = xs[v][0], y1s = ys[v][0], x2s = xs[v][1], y2s = ys[v][1];
            int x1d, y1d, x2d, y2d; dest_xy(x1s, y1s, m, x1d, y1d); dest_xy(x2s, y2s, m, x2d, y2d);
            int md = std::abs(x1d - x2d) + std::abs(y1d - y2d);
            return (md == 1) ? (md - 30) : md;
        }

        FINL void mark_once(const int16_t* stv, const RotMap& m, vector<int>& out) const {
            begin_eval();
            for (auto si16 : m.src) {
                int v = (int)stv[(int)si16];
                if ((unsigned)v >= (unsigned)MAXV || cnt[v] != 2) continue;
                uint32_t& t = seenStamp[v];
                if (t == epoch) continue;
                t = epoch; out.push_back(v);
            }
        }

        FINL int eval_after_rotate_full(const RotMap& m) const {
            const int16_t* stv = st->cells.data();
            vector<int> affected; affected.reserve(m.src.size());
            mark_once(stv, m, affected);
            int delta = 0;
            for (int v : affected) delta += pairCostAfter_full(m, v) - curCost[v];
            return base + delta;
        }

        FINL int eval_after_rotate_LB(const RotMap& m) const {
            const int16_t* stv = st->cells.data();
            vector<int> affected; affected.reserve(m.src.size());
            mark_once(stv, m, affected);
            int delta = 0;
            for (int v : affected) delta += pairCostAfter_LB(m, v) - curCost[v];
            return base + delta;
        }

        // ★ 軽量スコア: 園内に含まれる「非隣接ペア構成セル」の数（近傍優先）
        FINL int nonadj_in_square(const RotMap& m) const {
            int s = 0;
            for (uint16_t gi : m.src) s += nonAdjMask[(int)gi];
            return s;
        }
    };

    // I/O
    void loadJson(const string& path);

    // Search
    shared_ptr<Node_Beam> beamSearch(const Field& start, int beamWidth);

    // Utils
    int heuristic_box(const Field& f) const;
    int countAdjacentPairs(const Field& f) const;
    void compressSolutionSteps();                 // ★ 回転列の mod4 簡約
    void printSolution_write() const;
    void writeSummaryJson(const Field& finalField) const;

    // Infra
    void buildRotMaps(int N);
    void buildZobrist(int N, int maxVal /* = MAXV */);
    void buildRowBase(int N);

    Sig computeSig(const Field& f) const;
    Sig applyRotateAndSig(Field& st, const Sig& parent, const RotMap& m) const;

    FINL void rotate_apply(Field& st, const vector<uint16_t>& from, const vector<uint16_t>& to) const {
        const int k = (int)from.size();
        static thread_local vector<int16_t> buf; buf.resize(k);
        for (int t = 0; t < k; ++t) buf[t] = st.cells[from[t]];
        for (int t = 0; t < k; ++t) st.cells[to[t]] = buf[t];
    }
    FINL void rotateCW_inplace(Field& st, const RotMap& m)  const { rotate_apply(st, m.src, m.dst); }
    FINL void rotateCCW_inplace(Field& st, const RotMap& m) const { rotate_apply(st, m.dst, m.src); }

    // State
    Field field;
    json parsed;
    int fieldSize = 0;
    vector<tuple<int, int, int>> solutionSteps;

    vector<RotMap> rotMaps;
    vector<vector<uint64_t>> Z1, Z2;
    mutable std::mt19937_64 rng{ 0x9e3779b97f4a7c15ULL };
    vector<int> rowBase;

#ifdef USE_CUDA
    // GPU 用：RotMap の x/y/s をホスト側で保持（毎回再生成しない）
    vector<int> rotXs_host, rotYs_host, rotSs_host;

    // GPU 評価（1 ノード分の全回転を並列評価し、上位 K を返す）
    vector<Cand> evaluateNodeOnGPU(const shared_ptr<Node_Beam>& cur, int beamWidth);
#endif
};

App::App(const string& jsonPath) : field(0) {
    loadJson(jsonPath);
    fieldSize = parsed["problem"]["field"]["size"].get<int>();
    field = Field(fieldSize);
    for (int y = 0; y < fieldSize; ++y)
        for (int x = 0; x < fieldSize; ++x)
            field.at(y, x) = (int16_t)parsed["problem"]["field"]["entities"][y][x].get<int>();
    buildRotMaps(fieldSize);
    // ★ Zobrist を値域に合わせて縮小（N*N/2）
    buildZobrist(fieldSize, fieldSize * fieldSize / 2);
    buildRowBase(fieldSize);
}

void App::loadJson(const string& path) {
    ifstream ifs(path);
    if (!ifs) throw runtime_error("cannot open json file");
    ifs >> parsed;
    if (!parsed.contains("problem") || parsed["problem"].is_null())
        throw runtime_error("problem is null; not started or invalid");
}

void App::buildRotMaps(int N) {
    rotMaps.clear(); rotMaps.reserve(N * N * N);
#ifdef USE_CUDA
    rotXs_host.clear(); rotYs_host.clear(); rotSs_host.clear();
    rotXs_host.reserve(N * N * N);
    rotYs_host.reserve(N * N * N);
    rotSs_host.reserve(N * N * N);
#endif
    for (int s = 2; s <= N; ++s) {
        for (int y = 0; y + s <= N; ++y) {
            for (int x = 0; x + s <= N; ++x) {
                RotMap m; m.x = x; m.y = y; m.s = s;
                m.src.reserve(s * s); m.dst.reserve(s * s); m.dst2src.resize(s * s);
                for (int dy = 0; dy < s; ++dy) {
                    for (int dx = 0; dx < s; ++dx) {
                        uint16_t srcIdx = (uint16_t)((y + dy) * N + (x + dx));
                        int ndy = dx, ndx = s - 1 - dy;
                        uint16_t dstIdx = (uint16_t)((y + ndy) * N + (x + ndx));
                        m.src.push_back(srcIdx);
                        m.dst.push_back(dstIdx);
                        m.dst2src[ndy * s + ndx] = srcIdx;
                    }
                }
#ifdef USE_CUDA
                rotXs_host.push_back(x);
                rotYs_host.push_back(y);
                rotSs_host.push_back(s);
#endif
                rotMaps.push_back(std::move(m));
            }
        }
    }
}

void App::buildZobrist(int N, int maxVal) {
    Z1.assign(maxVal, vector<uint64_t>(N * N));
    Z2.assign(maxVal, vector<uint64_t>(N * N));
    for (int v = 0; v < maxVal; ++v)
        for (int i = 0; i < N * N; ++i) { Z1[v][i] = rng(); Z2[v][i] = rng(); }
}

void App::buildRowBase(int N) {
    rowBase.resize(N);
    for (int y = 0; y < N; ++y) rowBase[y] = y * N;
}

Sig App::computeSig(const Field& f) const {
    uint64_t h1 = 0, h2 = 0; const int NN = f.size * f.size;
    for (int i = 0; i < NN; ++i) {
        int v = (int)f.cells[i];
        if ((unsigned)v < (unsigned)Z1.size()) { h1 ^= Z1[v][i]; h2 ^= Z2[v][i]; }
    }
    return { h1,h2 };
}

Sig App::applyRotateAndSig(Field& st, const Sig& parent, const RotMap& m) const {
    const int k = (int)m.src.size();
    static thread_local vector<int16_t> buf; buf.resize(k);
    uint64_t h1 = parent.h1, h2 = parent.h2;
    for (int t = 0; t < k; ++t) { int i = (int)m.src[t]; int v = (int)st.cells[i]; buf[t] = (int16_t)v; h1 ^= Z1[v][i]; h2 ^= Z2[v][i]; }
    for (int t = 0; t < k; ++t) { int j = (int)m.dst[t]; int v = (int)buf[t]; st.cells[j] = (int16_t)v; h1 ^= Z1[v][j]; h2 ^= Z2[v][j]; }
    return { h1,h2 };
}

// ---- ヒューリスティック（初期/子ノードで使用）----
int App::heuristic_box(const Field& f) const {
    static thread_local uint8_t cnt[MAXV];
    static thread_local short xs[MAXV][2], ys[MAXV][2];
    memset(cnt, 0, sizeof(cnt));

    const int N = f.size;
    for (int y = 0; y < N; ++y) {
        const int base = y * N;
        for (int x = 0; x < N; ++x) {
            int v = (int)f.cells[base + x];
            if ((unsigned)v < (unsigned)MAXV) {
                uint8_t c = cnt[v]++; if (c < 2) { xs[v][c] = (short)x; ys[v][c] = (short)y; }
            }
        }
    }

    int distSum = 0, bonus = 0;
    for (int v = 0; v < MAXV; ++v) if (cnt[v] == 2) {
        int x1 = xs[v][0], y1 = ys[v][0], x2 = xs[v][1], y2 = ys[v][1];
        int dx = std::abs(x1 - x2), dy = std::abs(y1 - y2); distSum += dx + dy;
        if (dx == 0 && dy == 1) {
            bonus -= 20;
            if (x1 + 1 < N && f.at(y1, x1 + 1) == f.at(y2, x2 + 1)) bonus -= 10;
        }
        else if (dx == 1 && dy == 0) {
            bonus -= 20;
            if (y1 + 1 < N && f.at(y1 + 1, x1) == f.at(y2 + 1, x2)) bonus -= 10;
        }
    }
    return distSum + bonus;
}

// ---- 隣接ペア数（高速 O(N²））----
int App::countAdjacentPairs(const Field& f) const {
    int N = f.size, cnt = 0;
    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            int v = (int)f.at(y, x);
            if (x + 1 < N && v == f.at(y, x + 1)) ++cnt; // 横
            if (y + 1 < N && v == f.at(y + 1, x)) ++cnt; // 縦
        }
    }
    return cnt;
}

// 比較関数
static FINL bool by_f(const App::Cand& a, const App::Cand& b) { return a.f < b.f; }

// ---- CPU 版で使うローカルトップK（保持）----
struct LocalTopK {
    int K;
    vector<App::Cand> heap; // 最大ヒープ（悪い方K）
    explicit LocalTopK(int k = 0) :K(k) { heap.reserve(K); }
    FINL int worst() const { return (int)heap.size() < K ? INT_MAX : heap.front().f; }
    FINL void consider(App::Cand&& c) {
        auto comp = [](const App::Cand& a, const App::Cand& b) { return a.f < b.f; };
        if ((int)heap.size() < K) { heap.emplace_back(std::move(c)); if ((int)heap.size() == K) std::make_heap(heap.begin(), heap.end(), comp); }
        else if (c.f < heap.front().f) { std::pop_heap(heap.begin(), heap.end(), comp); heap.back() = std::move(c); std::push_heap(heap.begin(), heap.end(), comp); }
    }
    FINL void drain_to(vector<App::Cand>& out) {
        std::sort_heap(heap.begin(), heap.end(), [](const App::Cand& a, const App::Cand& b) { return a.f < b.f; });
        out.insert(out.end(), make_move_iterator(heap.begin()), make_move_iterator(heap.end()));
        heap.clear();
    }
};

#ifdef USE_CUDA
// ===============================
// ========== CUDA 部分 ==========
// ===============================

__device__ __forceinline__ bool inside_dev(int x, int y, int s, int xi, int yi) {
    return (unsigned)(xi - x) < (unsigned)s && (unsigned)(yi - y) < (unsigned)s;
}

__device__ __forceinline__ void dest_xy_dev(int x, int y, int s, int xs, int ys, int& xd, int& yd) {
    if (inside_dev(x, y, s, xs, ys)) {
        xd = x + (s - 1 - (ys - y));
        yd = y + (xs - x);
    } else {
        xd = xs; yd = ys;
    }
}

__device__ __forceinline__ int val_after_at_dev(int N, const int16_t* cells, int x, int y, int s, int xd, int yd) {
    if (inside_dev(x, y, s, xd, yd)) {
        int xs = x + (yd - y);
        int ys = y + (s - 1 - (xd - x));
        return (int)cells[ys * N + xs];
    } else {
        return (int)cells[yd * N + xd];
    }
}

__device__ __forceinline__ int pairCostAfter_full_dev(
    int N, const int16_t* cells,
    int x, int y, int s,
    int x1s, int y1s, int x2s, int y2s)
{
    int x1d, y1d, x2d, y2d;
    dest_xy_dev(x, y, s, x1s, y1s, x1d, y1d);
    dest_xy_dev(x, y, s, x2s, y2s, x2d, y2d);
    int dx = abs(x1d - x2d), dy = abs(y1d - y2d);
    int md = dx + dy; int cost = md;
    if (md == 1) {
        if (x1d == x2d) {
            if (x1d + 1 < N) {
                int r1 = val_after_at_dev(N, cells, x, y, s, x1d + 1, y1d);
                int r2 = val_after_at_dev(N, cells, x, y, s, x2d + 1, y2d);
                if (r1 == r2) cost -= 10;
            }
            cost -= 20;
        } else {
            if (y1d + 1 < N) {
                int d1 = val_after_at_dev(N, cells, x, y, s, x1d, y1d + 1);
                int d2 = val_after_at_dev(N, cells, x, y, s, x2d, y2d + 1);
                if (d1 == d2) cost -= 10;
            }
            cost -= 20;
        }
    }
    return cost;
}

__device__ __forceinline__ int pairCostAfter_LB_dev(
    int x, int y, int s,
    int x1s, int y1s, int x2s, int y2s)
{
    int x1d, y1d, x2d, y2d;
    dest_xy_dev(x, y, s, x1s, y1s, x1d, y1d);
    dest_xy_dev(x, y, s, x2s, y2s, x2d, y2d);
    int md = abs(x1d - x2d) + abs(y1d - y2d);
    return (md == 1) ? (md - 30) : md;
}

__global__ void eval_rotations_kernel(
    int N, int numMaps, int g_next, int base,
    const int16_t* cells,
    const uint8_t* cnt,
    const short* xspos, const short* yspos,
    const int* curCost,
    const uint8_t* nonAdjMask,
    const int* rotX, const int* rotY, const int* rotS,
    int* out_f, int* out_h)
{
    int rid = blockIdx.x * blockDim.x + threadIdx.x;
    if (rid >= numMaps) return;

    int x = rotX[rid], y = rotY[rid], s = rotS[rid];

    bool affected[MAXV];
#pragma unroll
    for (int v = 0; v < MAXV; ++v) affected[v] = false;

    for (int dy = 0; dy < s; ++dy) {
        int yy = y + dy;
        int rowBase = yy * N;
        for (int dx = 0; dx < s; ++dx) {
            int xx = x + dx;
            int v = (int)cells[rowBase + xx];
            if ((unsigned)v < (unsigned)MAXV && cnt[v] == 2) affected[v] = true;
        }
    }

    int nonadjCount = 0;
    for (int dy = 0; dy < s; ++dy) {
        int baseRow = (y + dy) * N;
        for (int dx = 0; dx < s; ++dx) {
            nonadjCount += nonAdjMask[baseRow + (x + dx)];
        }
    }

    int delta_full = 0;
    int delta_lb   = 0;
    for (int v = 0; v < MAXV; ++v) if (affected[v]) {
        int x1s = xspos[v * 2 + 0];
        int y1s = yspos[v * 2 + 0];
        int x2s = xspos[v * 2 + 1];
        int y2s = yspos[v * 2 + 1];
        int c_full = pairCostAfter_full_dev(N, cells, x, y, s, x1s, y1s, x2s, y2s);
        int c_lb   = pairCostAfter_LB_dev  (x, y, s, x1s, y1s, x2s, y2s);
        delta_full += c_full - curCost[v];
        delta_lb   += c_lb   - curCost[v];
    }

    int h   = base + delta_full;
    int hLB = base + delta_lb;
    (void)hLB;

    // prSmall/prNear は gate 用だが、GPU 版では全面評価 → ホストで TopK 選抜
    int f = g_next + h;
    (void)nonadjCount;

    out_f[rid] = f;
    out_h[rid] = h;
}

vector<App::Cand> App::evaluateNodeOnGPU(const shared_ptr<Node_Beam>& cur, int beamWidth) {
    const int N = cur->state.size;
    const int numMaps = (int)rotMaps.size();

    // --- HeuCtx をホストで構築して必要配列を平坦化 ---
    HeuCtx ctx(cur->state, rowBase);
    vector<uint8_t> cnt_host = ctx.cnt; // MAXV
    vector<short> xs_host(MAXV * 2, 0), ys_host(MAXV * 2, 0);
    vector<int>   curCost_host(MAXV, 0);
    for (int v = 0; v < MAXV; ++v) {
        if (ctx.cnt[v] >= 1) { xs_host[v * 2 + 0] = ctx.xs[v][0]; ys_host[v * 2 + 0] = ctx.ys[v][0]; }
        if (ctx.cnt[v] >= 2) { xs_host[v * 2 + 1] = ctx.xs[v][1]; ys_host[v * 2 + 1] = ctx.ys[v][1]; }
        curCost_host[v] = ctx.curCost[v];
    }
    const int base = ctx.base;

    // --- デバイスメモリ確保 ---
    int16_t* d_cells = nullptr;
    uint8_t* d_cnt = nullptr; short* d_xs = nullptr; short* d_ys = nullptr;
    int* d_curCost = nullptr; uint8_t* d_nonAdj = nullptr;
    int* d_rx = nullptr; int* d_ry = nullptr; int* d_rs = nullptr;
    int* d_out_f = nullptr; int* d_out_h = nullptr;

    CUDA_CHECK(cudaMalloc(&d_cells,  N * N * sizeof(int16_t)));
    CUDA_CHECK(cudaMalloc(&d_cnt,    MAXV * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_xs,     MAXV * 2 * sizeof(short)));
    CUDA_CHECK(cudaMalloc(&d_ys,     MAXV * 2 * sizeof(short)));
    CUDA_CHECK(cudaMalloc(&d_curCost,MAXV * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nonAdj, N * N * sizeof(uint8_t)));

    CUDA_CHECK(cudaMalloc(&d_rx, numMaps * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ry, numMaps * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rs, numMaps * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&d_out_f, numMaps * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out_h, numMaps * sizeof(int)));

    // --- 転送 ---
    CUDA_CHECK(cudaMemcpy(d_cells,  cur->state.cells.data(), N * N * sizeof(int16_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cnt,    cnt_host.data(),         MAXV * sizeof(uint8_t),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_xs,     xs_host.data(),          MAXV * 2 * sizeof(short),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ys,     ys_host.data(),          MAXV * 2 * sizeof(short),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_curCost,curCost_host.data(),     MAXV * sizeof(int),      cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nonAdj, ctx.nonAdjMask.data(),   N * N * sizeof(uint8_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_rx, rotXs_host.data(), numMaps * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ry, rotYs_host.data(), numMaps * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rs, rotSs_host.data(), numMaps * sizeof(int), cudaMemcpyHostToDevice));

    // --- 起動 ---
    int g_next = cur->g + 1;
    dim3 block(256);
    dim3 grid((numMaps + block.x - 1) / block.x);
    eval_rotations_kernel<<<grid, block>>>(
        N, numMaps, g_next, base,
        d_cells, d_cnt, d_xs, d_ys, d_curCost,
        d_nonAdj, d_rx, d_ry, d_rs,
        d_out_f, d_out_h);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- 取得 ---
    vector<int> f_host(numMaps), h_host(numMaps);
    CUDA_CHECK(cudaMemcpy(f_host.data(), d_out_f, numMaps * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_host.data(), d_out_h, numMaps * sizeof(int), cudaMemcpyDeviceToHost));

    // --- Top-K 選抜（ホスト） ---
    vector<int> idx(numMaps);
    std::iota(idx.begin(), idx.end(), 0);
    if ((int)idx.size() > beamWidth) {
        std::nth_element(idx.begin(), idx.begin() + beamWidth, idx.end(),
            [&](int a, int b) { return f_host[a] < f_host[b]; });
        idx.resize(beamWidth);
    }
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return f_host[a] < f_host[b]; });

    vector<Cand> out;
    out.reserve(idx.size());
    for (int rid : idx) {
        Cand c;
        c.g  = g_next;
        c.h  = h_host[rid];
        c.f  = f_host[rid];
        c.rx = rotXs_host[rid];
        c.ry = rotYs_host[rid];
        c.rs = rotSs_host[rid];
        c.parent = cur;
        c.map = &rotMaps[rid];
        out.push_back(std::move(c));
    }

    // --- 後片付け ---
    cudaFree(d_cells);  cudaFree(d_cnt);  cudaFree(d_xs);  cudaFree(d_ys);
    cudaFree(d_curCost);cudaFree(d_nonAdj);
    cudaFree(d_rx); cudaFree(d_ry); cudaFree(d_rs);
    cudaFree(d_out_f); cudaFree(d_out_h);

    return out;
}
#endif // USE_CUDA

// ---- Beam 検索 ----
shared_ptr<App::Node_Beam> App::beamSearch(const Field& start, int beamWidth) {
    auto root = make_shared<Node_Beam>(fieldSize);
    root->state = start; root->g = 0; root->h = heuristic_box(start); root->sig = computeSig(start);

    vector<shared_ptr<Node_Beam>> beam = { root };
    FastHashSet<Sig, SigHash> visited; visited.reserve(200000);
#if !defined(ankerl_unordered_dense_h) && !defined(ABSL_CONTAINER_FLAT_HASH_SET_H_)
    visited.max_load_factor(0.7f);
#endif

    // ★ h最小・実ペア数最大の両建てで最良を追跡
    shared_ptr<Node_Beam> bestNodeByH = root;
    int bestH = root->h;
    shared_ptr<Node_Beam> bestNodeByPairs = root;
    int bestPairs = countAdjacentPairs(root->state);

    int noImp = 0, maxNoImp = 100;

#ifdef _OPENMP
    const int T = max(1, omp_get_max_threads());
#else
    const int T = 1;
#endif

    while (!beam.empty()) {
        // dedup
        vector<shared_ptr<Node_Beam>> uniq; uniq.reserve(beam.size());
        for (auto& cur : beam) if (visited.insert(cur->sig).second) uniq.push_back(cur);
        if (uniq.empty()) break;

        vector<Cand> pool;

#ifdef USE_CUDA
        // --- GPU: 各ノードを順に GPU で評価し、ノード内の TopK を pool に集約 ---
        for (auto& cur : uniq) {
            auto candTop = evaluateNodeOnGPU(cur, beamWidth);
            pool.insert(pool.end(), make_move_iterator(candTop.begin()), make_move_iterator(candTop.end()));
        }
#else
        // --- CPU(OpenMP) 旧実装 ---
        vector<LocalTopK> localTop(T, LocalTopK(beamWidth));

#ifdef _OPENMP
        if ((int)uniq.size() == 1 && T > 1) {
            auto& cur = uniq[0];
#pragma omp parallel
            {
                int tid = omp_get_thread_num();
                HeuCtx ctx(cur->state, rowBase);
#pragma omp for schedule(static)
                for (int rid = 0; rid < (int)rotMaps.size(); ++rid) {
                    const RotMap& Mref = rotMaps[rid];
                    int g = cur->g + 1;

                    // 下界
                    int h_lb = ctx.eval_after_rotate_LB(Mref);

                    // ★ 小園・近傍 優先（順序付けヒント）
                    int prSmall = (Mref.s <= 4 ? 5 - Mref.s : 0);          // s=2:3, s=3:2, s=4:1
                    int prNear = std::min(8, ctx.nonadj_in_square(Mref)); // 上限8
                    int gate = g + h_lb - (prSmall + prNear);
                    if (gate >= localTop[tid].worst()) continue;

                    // 完全版
                    int h = ctx.eval_after_rotate_full(Mref);
                    int f = g + h;
                    if (f < localTop[tid].worst()) {
                        Cand c; c.g = g; c.h = h; c.f = f; c.rx = Mref.x; c.ry = Mref.y; c.rs = Mref.s; c.parent = cur; c.map = &Mref;
                        localTop[tid].consider(std::move(c));
                    }
                }
            }
        }
        else
#endif
        {
#ifdef _OPENMP
#pragma omp parallel for schedule(guided,8)
#endif
            for (int i = 0; i < (int)uniq.size(); ++i) {
#ifdef _OPENMP
                int tid = omp_get_thread_num();
#else
                int tid = 0;
#endif
                auto& cur = uniq[i];
                HeuCtx ctx(cur->state, rowBase);
                for (const RotMap& Mref : rotMaps) {
                    int g = cur->g + 1;
                    int h_lb = ctx.eval_after_rotate_LB(Mref);

                    int prSmall = (Mref.s <= 4 ? 5 - Mref.s : 0);
                    int prNear = std::min(8, ctx.nonadj_in_square(Mref));
                    int gate = g + h_lb - (prSmall + prNear);
                    if (gate >= localTop[tid].worst()) continue;

                    int h = ctx.eval_after_rotate_full(Mref);
                    int f = g + h;
                    if (f < localTop[tid].worst()) {
                        Cand c; c.g = g; c.h = h; c.f = f; c.rx = Mref.x; c.ry = Mref.y; c.rs = Mref.s; c.parent = cur; c.map = &Mref;
                        localTop[tid].consider(std::move(c));
                    }
                }
            }
        }

        // gather
        size_t est = 0; for (auto& lt : localTop) est += min(lt.K, (int)lt.heap.size());
        pool.reserve(est); for (auto& lt : localTop) lt.drain_to(pool);
#endif // USE_CUDA

        if (pool.empty()) break;

        if (pool.size() > (size_t)beamWidth) {
            std::nth_element(pool.begin(), pool.begin() + beamWidth, pool.end(), by_f);
            pool.resize(beamWidth);
        }
        std::sort(pool.begin(), pool.end(), by_f);

        // materialize
        vector<shared_ptr<Node_Beam>> nextBeam; nextBeam.reserve(pool.size());
        for (const auto& c : pool) {
            auto child = make_shared<Node_Beam>(fieldSize);
            child->state = c.parent->state;
            child->sig = applyRotateAndSig(child->state, c.parent->sig, *c.map);
            child->g = c.g; child->h = c.h;
            child->rx = c.rx; child->ry = c.ry; child->rs = c.rs;
            child->parent = c.parent;
            nextBeam.push_back(std::move(child));
        }

        // ★ h最小の正しい更新（front だけを見ない）
        auto itH = std::min_element(nextBeam.begin(), nextBeam.end(),
            [](const auto& a, const auto& b) { return a->h < b->h; });
        int curBestH = (*itH)->h;
        if (curBestH < bestH) { bestH = curBestH; noImp = 0; bestNodeByH = *itH; }
        else { if (++noImp >= maxNoImp) break; }

        // ★ 実ペア数ベスト（同数なら手数 g 少）も更新
        for (const auto& ch : nextBeam) {
            int pairs = countAdjacentPairs(ch->state);
            if (pairs > bestPairs || (pairs == bestPairs && ch->g < bestNodeByPairs->g)) {
                bestPairs = pairs; bestNodeByPairs = ch;
            }
        }

        beam = std::move(nextBeam);
    }

    // reconstruct solution（★ ペア数ベストの経路を採用）
    solutionSteps.clear();
    auto p = bestNodeByPairs;
    while (p->parent) { solutionSteps.emplace_back(p->rx, p->ry, p->rs); p = p->parent; }
    reverse(solutionSteps.begin(), solutionSteps.end());

    // ★ 連続同一園の回転を mod4 簡約
    compressSolutionSteps();

    return bestNodeByPairs;
}

// ---- 回転列の事後圧縮（同一園の連続回転を mod4 で簡約）----
void App::compressSolutionSteps() {
    vector<tuple<int, int, int>> out;
    const int n = (int)solutionSteps.size();
    for (int i = 0; i < n; ) {
        auto [x, y, s] = solutionSteps[i];
        int j = i + 1;
        while (j < n) {
            auto [x2, y2, s2] = solutionSteps[j];
            if (x2 == x && y2 == y && s2 == s) ++j; else break;
        }
        int run = j - i;
        int keep = run % 4; // 4回転は恒等
        for (int k = 0; k < keep; ++k) out.emplace_back(x, y, s);
        i = j;
    }
    solutionSteps.swap(out);
}

// ---- 出力 ----
void App::printSolution_write() const {
    json out; out["ops"] = json::array();
    for (auto& t : solutionSteps) {
        int x, y, n; tie(x, y, n) = t; out["ops"].push_back({ {"x",x},{"y",y},{"n",n} });
    }
    ofstream ofs("result.json"); ofs << out.dump(2);
}
void App::writeSummaryJson(const Field& finalField) const {
    json summary;
    summary["count"] = countAdjacentPairs(finalField);
    summary["steps"] = (int)solutionSteps.size();
    ofstream ofs("summary.json"); ofs << summary.dump(2);
}

// ---- メイン（通信系は残す）----
void App::run() {
    auto finalNode = beamSearch(field, 1000);

    if (solutionSteps.empty()) {
        cout << "解が見つかりませんでした。\n";
    }
    else {
        printSolution_write();
        Field finalField = finalNode->state;
        writeSummaryJson(finalField);
        cout << "result.json " << solutionSteps.size() << " \n";

        // 送信（必要なら cmd の system を有効化）
        const char post[] =
            "curl -X POST http://10.30.6.194:3000/submit -F \"result=@result.json\" -F \"numnum=@summary.json\"";
        cmd(post);
    }
}

int main() {
    // 取得（必要なら cmd の system を有効化）
    const char get[] = "curl http://10.30.6.194:3000/get_json -o match.json";
    App::cmd(get);

    try {
        App app("match.json");
        app.run();
    }
    catch (const exception& e) {
        cerr << e.what() << "\n";
        return -1;
    }
    return 0;
}
