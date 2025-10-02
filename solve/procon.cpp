// procon.cpp

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <nlohmann/json.hpp>
using json = nlohmann::json;
using namespace std;

#if defined(_MSC_VER)
#define FINL __forceinline
#else
#define FINL __attribute__((always_inline)) inline
#endif

// 定数
static const int MAXV = 300; // N<=24 なら N*N/2=288 以下

// 枝刈りパラメータ（控えめ）
static const int ASP_MARGIN = 8;
static const int LARGE_S_SKIP_THRESHOLD = 6;

// 盤面
struct Field {
    vector<int16_t> cells;
    int size;
    Field(int s = 0) : cells(s * s), size(s) {}
    FINL int16_t& at(int y, int x) { return cells[y * size + x]; }
    FINL int16_t  at(int y, int x) const { return cells[y * size + x]; }
};

// 回転マップ
struct RotMap {
    int x = 0, y = 0, s = 0;
    vector<uint16_t> src, dst;   // src[t] -> dst[t]
    vector<uint16_t> dst2src;    // 目的地オフセット -> 元index
};

// Zobrist
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

// 署名集合（環境に応じて速い実装が使われる）
#if defined(__has_include)
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
#else
  #include <unordered_set>
  template <class K, class H> using FastHashSet = std::unordered_set<K, H>;
#endif

class App {
public:
    App(const string& jsonPath);
    void run();

    // ビーム探索ノード
    struct Node_Beam {
        Field state;
        int g = 0, h = 0;
        int rx = 0, ry = 0, rs = 0;
        Sig sig{};
        // 固定化済み外周を除いた有効領域
        int leftBound = 0, topBound = 0, rightBound = 0, bottomBound = 0;

        shared_ptr<Node_Beam> parent;
        FINL int f() const { return g + h; }
        explicit Node_Beam(int size) : state(size) {}
    };

    // 外部コマンド
    static inline void cmd(const char cmd[]) {
        std::system(cmd);
    }

private:
    // ヒューリスティック用コンテキスト
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

        // 非隣接ペアの構成セルフラグ
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
                } else {
                    curCost[v] = 0;
                }
            }

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
            if (inside(m, xs_, ys_)) {
                xd = m.x + (m.s - 1 - (ys_ - m.y));
                yd = m.y + (xs_ - m.x);
            } else { xd = xs_; yd = ys_; }
        }
        FINL int val_after_at(const RotMap& m, int xd, int yd) const {
            if (inside(m, xd, yd)) {
                int off = (yd - m.y) * m.s + (xd - m.x);
                return (int)st->cells[(int)m.dst2src[off]];
            }
            return (int)st->cells[(*rowBase)[yd] + xd];
        }

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
            } else if (dx == 1 && dy == 0) {
                if (y1 + 1 < N && y2 + 1 < N) {
                    int d1 = (int)st->cells[(*rowBase)[y1 + 1] + x1];
                    int d2 = (int)st->cells[(*rowBase)[y2 + 1] + x2];
                    if (d1 == d2) cost -= 10;
                }
                cost -= 20;
            }
            return cost;
        }

        FINL int pairCostAfter_full(const RotMap& m, int v) const {
            int x1s = xs[v][0], y1s = ys[v][0], x2s = xs[v][1], y2s = ys[v][1];
            int x1d, y1d, x2d, y2d; dest_xy(x1s, y1s, m, x1d, y1d); dest_xy(x2s, y2s, m, x2d, y2d);
            int dx = std::abs(x1d - x2d), dy = std::abs(y1d - y2d); int md = dx + dy; int cost = md;
            if (md == 1) {
                if (x1d == x2d) { // 縦
                    if (x1d + 1 < N) {
                        bool in1 = inside(m, x1d + 1, y1d);
                        bool in2 = inside(m, x2d + 1, y2d);
                        int r1 = in1 ? (int)st->cells[(int)m.dst2src[(y1d - m.y) * m.s + ((x1d + 1) - m.x)]]
                                     : (int)st->cells[(*rowBase)[y1d] + (x1d + 1)];
                        int r2 = in2 ? (int)st->cells[(int)m.dst2src[(y2d - m.y) * m.s + ((x2d + 1) - m.x)]]
                                     : (int)st->cells[(*rowBase)[y2d] + (x2d + 1)];
                        if (r1 == r2) cost -= 10;
                    }
                    cost -= 20;
                } else {          // 横
                    if (y1d + 1 < N) {
                        bool in1 = inside(m, x1d, y1d + 1);
                        bool in2 = inside(m, x2d, y2d + 1);
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

        FINL int eval_after_rotate_full_with_list(const RotMap& m, const vector<int>& affected) const {
            int delta = 0;
            for (int v : affected) delta += pairCostAfter_full(m, v) - curCost[v];
            return base + delta;
        }
        FINL int eval_after_rotate_LB_with_list(const RotMap& m, const vector<int>& affected) const {
            int delta = 0;
            for (int v : affected) delta += pairCostAfter_LB(m, v) - curCost[v];
            return base + delta;
        }

        // 回転領域内の非隣接セル数（簡易優先度）
        FINL int nonadj_in_square(const RotMap& m) const {
            int s = 0;
            for (uint16_t gi : m.src) s += nonAdjMask[(int)gi];
            return s;
        }
    };

    // I/O
    void loadJson(const string& path);

    // 探索
    shared_ptr<Node_Beam> beamSearch(const Field& start, int beamWidth);

    // ユーティリティ
    int heuristic_box(const Field& f) const;
    int countAdjacentPairs(const Field& f) const;
    void compressSolutionSteps();
    void printSolution_write() const;
    void writeSummaryJson(const Field& finalField) const;

    // 準備
    void buildRotMaps(int N);
    void buildZobrist(int N, int maxVal);
    void buildRowBase(int N);

    Sig computeSig(const Field& f) const;
    Sig applyRotateAndSig(Field& st, const Sig& parent, const RotMap& m) const;

    // 状態を変えずに子の署名だけ計算
    Sig sigAfterRotateNoApply(const Field& st, const Sig& parent, const RotMap& m) const;

    // 実質的な無変化回転の判定
    bool isRotationNoop(const Field& st, const RotMap& m) const;

    FINL void rotate_apply(Field& st, const vector<uint16_t>& from, const vector<uint16_t>& to) const {
        const int k = (int)from.size();
        static thread_local vector<int16_t> buf; buf.resize(k);
        for (int t = 0; t < k; ++t) buf[t] = st.cells[from[t]];
        for (int t = 0; t < k; ++t) st.cells[to[t]] = buf[t];
    }
    FINL void rotateCW_inplace(Field& st, const RotMap& m)  const { rotate_apply(st, m.src, m.dst); }
    FINL void rotateCCW_inplace(Field& st, const RotMap& m) const { rotate_apply(st, m.dst, m.src); }

    // 固定化系
    bool rowsIdentical(const Field& f, int y1, int y2, int left, int right) const;
    bool rowHasHorizontalPairs(const Field& f, int y, int left, int right) const;
    bool colsIdentical(const Field& f, int x1, int x2, int top, int bottom) const;
    bool colHasVerticalPairs(const Field& f, int x, int top, int bottom) const;

    // 2行/2列の混合固定判定
    bool canFixTopTwoRowsMix(const Field& f, int left, int right, int top) const;
    bool canFixBottomTwoRowsMix(const Field& f, int left, int right, int bottom) const;
    bool canFixLeftTwoColsMix(const Field& f, int top, int bottom, int left) const;
    bool canFixRightTwoColsMix(const Field& f, int top, int bottom, int right) const;

    // 境界の縮小
    void applyFixationLoop(const Field& f, int& left, int& top, int& right, int& bottom) const;

    // 状態
    Field field;
    json parsed;
    int fieldSize = 0;
    vector<tuple<int, int, int>> solutionSteps;

    vector<RotMap> rotMaps;
    vector<vector<uint64_t>> Z1, Z2;
    mutable std::mt19937_64 rng{ 0x9e3779b97f4a7c15ULL };
    vector<int> rowBase;
};

App::App(const string& jsonPath) : field(0) {
    loadJson(jsonPath);
    fieldSize = parsed["problem"]["field"]["size"].get<int>();
    field = Field(fieldSize);
    for (int y = 0; y < fieldSize; ++y)
        for (int x = 0; x < fieldSize; ++x)
            field.at(y, x) = (int16_t)parsed["problem"]["field"]["entities"][y][x].get<int>();
    buildRotMaps(fieldSize);
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
    size_t zv = Z1.size();
    for (int i = 0; i < NN; ++i) {
        int v = (int)f.cells[i];
        if ((unsigned)v < (unsigned)zv) { h1 ^= Z1[v][i]; h2 ^= Z2[v][i]; }
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

Sig App::sigAfterRotateNoApply(const Field& st, const Sig& parent, const RotMap& m) const {
    uint64_t h1 = parent.h1, h2 = parent.h2;
    const int16_t* cells = st.cells.data();
    const int k = (int)m.src.size();
    for (int t = 0; t < k; ++t) {
        int i = (int)m.src[t];
        int j = (int)m.dst[t];
        int v = (int)cells[i];
        h1 ^= Z1[v][i]; h2 ^= Z2[v][i];
        h1 ^= Z1[v][j]; h2 ^= Z2[v][j];
    }
    return { h1, h2 };
}

bool App::isRotationNoop(const Field& st, const RotMap& m) const {
    const int16_t* cells = st.cells.data();
    const int k = (int)m.src.size();
    for (int t = 0; t < k; ++t) {
        if (cells[(int)m.src[t]] != cells[(int)m.dst[t]]) return false;
    }
    return true;
}

// 初期ヒューリスティック
int App::heuristic_box(const Field& f) const {
    static thread_local uint8_t cnt[MAXV];
    static thread_local short xs[MAXV][2], ys[MAXV][2];
    std::memset(cnt, 0, sizeof(cnt));

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
        } else if (dx == 1 && dy == 0) {
            bonus -= 20;
            if (y1 + 1 < N && f.at(y1 + 1, x1) == f.at(y2 + 1, x2)) bonus -= 10;
        }
    }
    return distSum + bonus;
}

// 隣接ペア数
int App::countAdjacentPairs(const Field& f) const {
    int N = f.size, cnt = 0;
    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            int v = (int)f.at(y, x);
            if (x + 1 < N && v == f.at(y, x + 1)) ++cnt;
            if (y + 1 < N && v == f.at(y + 1, x)) ++cnt;
        }
    }
    return cnt;
}

// 固定化系ヘルパ
bool App::rowsIdentical(const Field& f, int y1, int y2, int left, int right) const {
    if (y1 < 0 || y2 < 0 || y1 >= f.size || y2 >= f.size) return false;
    for (int x = left; x < right; ++x) if (f.at(y1, x) != f.at(y2, x)) return false;
    return true;
}
bool App::rowHasHorizontalPairs(const Field& f, int y, int left, int right) const {
    if (y < 0 || y >= f.size) return false;
    int w = right - left;
    if ((w & 1) != 0) return false;
    for (int x = left; x < right; x += 2) if (f.at(y, x) != f.at(y, x + 1)) return false;
    return true;
}
bool App::colsIdentical(const Field& f, int x1, int x2, int top, int bottom) const {
    if (x1 < 0 || x2 < 0 || x1 >= f.size || x2 >= f.size) return false;
    for (int y = top; y < bottom; ++y) if (f.at(y, x1) != f.at(y, x2)) return false;
    return true;
}
bool App::colHasVerticalPairs(const Field& f, int x, int top, int bottom) const {
    if (x < 0 || x >= f.size) return false;
    int h = bottom - top;
    if ((h & 1) != 0) return false;
    for (int y = top; y < bottom; y += 2) if (f.at(y, x) != f.at(y + 1, x)) return false;
    return true;
}

// 2行/2列の混合固定判定
bool App::canFixTopTwoRowsMix(const Field& f, int left, int right, int top) const {
    if (top + 1 >= f.size) return false;
    int x = left;
    while (x < right) {
        int a0 = f.at(top, x);
        int a1 = f.at(top + 1, x);
        if (a0 == a1) { ++x; continue; } // 縦ペア1列
        if (x + 1 < right) {             // 2列横ペア
            int b0 = f.at(top, x + 1);
            int b1 = f.at(top + 1, x + 1);
            if (a0 == b0 && a1 == b1) { x += 2; continue; }
        }
        return false;
    }
    return true;
}
bool App::canFixBottomTwoRowsMix(const Field& f, int left, int right, int bottom) const {
    if (bottom < 2) return false;
    return canFixTopTwoRowsMix(f, left, right, bottom - 2);
}
bool App::canFixLeftTwoColsMix(const Field& f, int top, int bottom, int left) const {
    if (left + 1 >= f.size) return false;
    int y = top;
    while (y < bottom) {
        int a0 = f.at(y, left);
        int a1 = f.at(y, left + 1);
        if (a0 == a1) { ++y; continue; } // 横ペア1行
        if (y + 1 < bottom) {            // 2行縦ペア
            int b0 = f.at(y + 1, left);
            int b1 = f.at(y + 1, left + 1);
            if (a0 == b0 && a1 == b1) { y += 2; continue; }
        }
        return false;
    }
    return true;
}
bool App::canFixRightTwoColsMix(const Field& f, int top, int bottom, int right) const {
    if (right < 2) return false;
    return canFixLeftTwoColsMix(f, top, bottom, right - 2);
}

// 外周から縮める
void App::applyFixationLoop(const Field& f, int& left, int& top, int& right, int& bottom) const {
    const int N = f.size;
    left   = std::max(0, std::min(left,   N));
    right  = std::max(0, std::min(right,  N));
    top    = std::max(0, std::min(top,    N));
    bottom = std::max(0, std::min(bottom, N));

    bool changed = true;
    while (changed && left < right && top < bottom) {
        changed = false;

        // 上
        if (top + 1 < bottom && canFixTopTwoRowsMix(f, left, right, top)) {
            top += 2; changed = true; continue;
        }
        if (top + 1 < bottom && rowsIdentical(f, top, top + 1, left, right)) {
            top += 2; changed = true; continue;
        }
        if (top < bottom && rowHasHorizontalPairs(f, top, left, right)) {
            top += 1; changed = true; continue;
        }

        // 下
        if (top + 1 < bottom && canFixBottomTwoRowsMix(f, left, right, bottom)) {
            bottom -= 2; changed = true; continue;
        }
        if (top + 1 < bottom && rowsIdentical(f, bottom - 2, bottom - 1, left, right)) {
            bottom -= 2; changed = true; continue;
        }
        if (top < bottom && rowHasHorizontalPairs(f, bottom - 1, left, right)) {
            bottom -= 1; changed = true; continue;
        }

        // 左
        if (left + 1 < right && canFixLeftTwoColsMix(f, top, bottom, left)) {
            left += 2; changed = true; continue;
        }
        if (left + 1 < right && colsIdentical(f, left, left + 1, top, bottom)) {
            left += 2; changed = true; continue;
        }
        if (left < right && colHasVerticalPairs(f, left, top, bottom)) {
            left += 1; changed = true; continue;
        }

        // 右
        if (left + 1 < right && canFixRightTwoColsMix(f, top, bottom, right)) {
            right -= 2; changed = true; continue;
        }
        if (left + 1 < right && colsIdentical(f, right - 2, right - 1, top, bottom)) {
            right -= 2; changed = true; continue;
        }
        if (left < right && colHasVerticalPairs(f, right - 1, top, bottom)) {
            right -= 1; changed = true; continue;
        }
    }
}

// ビーム探索
struct Cand {
    int f, h, g;
    int rx, ry, rs;
    shared_ptr<App::Node_Beam> parent;
    const RotMap* map;
    Sig sig;
};
static FINL bool by_f(const Cand& a, const Cand& b) { return a.f < b.f; }

struct LocalTopK {
    int K;
    vector<Cand> heap; // 最大ヒープ（悪い方K）
    explicit LocalTopK(int k = 0) :K(k) { heap.reserve(K); }
    FINL int worst() const { return (int)heap.size() < K ? std::numeric_limits<int>::max() : heap.front().f; }
    FINL void consider(Cand&& c) {
        auto comp = [](const Cand& a, const Cand& b) { return a.f < b.f; };
        if ((int)heap.size() < K) { heap.emplace_back(std::move(c)); if ((int)heap.size() == K) std::make_heap(heap.begin(), heap.end(), comp); }
        else if (c.f < heap.front().f) { std::pop_heap(heap.begin(), heap.end(), comp); heap.back() = std::move(c); std::push_heap(heap.begin(), heap.end(), comp); }
    }
    FINL void drain_to(vector<Cand>& out) {
        std::sort_heap(heap.begin(), heap.end(), [](const Cand& a, const Cand& b) { return a.f < b.f; });
        out.insert(out.end(), make_move_iterator(heap.begin()), make_move_iterator(heap.end()));
        heap.clear();
    }
};

shared_ptr<App::Node_Beam> App::beamSearch(const Field& start, int beamWidth) {
    auto root = make_shared<Node_Beam>(fieldSize);
    root->state = start; root->g = 0; root->h = heuristic_box(start); root->sig = computeSig(start);
    root->leftBound = 0; root->topBound = 0; root->rightBound = fieldSize; root->bottomBound = fieldSize;
    applyFixationLoop(root->state, root->leftBound, root->topBound, root->rightBound, root->bottomBound);

    vector<shared_ptr<Node_Beam>> beam = { root };
    FastHashSet<Sig, SigHash> visited; visited.reserve(200000);
#if !defined(ankerl_unordered_dense_h) && !defined(ABSL_CONTAINER_FLAT_HASH_SET_H_)
    if constexpr (std::is_same<FastHashSet<Sig, SigHash>, std::unordered_set<Sig, SigHash>>::value) {
        visited.max_load_factor(0.7f);
    }
#endif

    shared_ptr<Node_Beam> bestNodeByH = root;
    int bestH = root->h;
    shared_ptr<Node_Beam> bestNodeByPairs = root;
    int bestPairs = countAdjacentPairs(root->state);

    int noImp = 0, maxNoImp = 100;

#ifdef _OPENMP
    const int T = std::max(1, omp_get_max_threads());
#else
    const int T = 1;
#endif

    while (!beam.empty()) {
        // 同層重複除去
        vector<shared_ptr<Node_Beam>> uniq; uniq.reserve(beam.size());
        for (auto& cur : beam) if (visited.insert(cur->sig).second) uniq.push_back(cur);
        if (uniq.empty()) break;

        // アスピレーション
        int beamBestF = std::numeric_limits<int>::max();
        for (auto& n : uniq) beamBestF = std::min(beamBestF, n->f());
        const int aspCutGlobal = (beamBestF == std::numeric_limits<int>::max() ? beamBestF : beamBestF + ASP_MARGIN);

        vector<LocalTopK> localTop(T, LocalTopK(beamWidth));

#ifdef _OPENMP
        if ((int)uniq.size() == 1 && T > 1) {
            auto& cur = uniq[0];
#pragma omp parallel
            {
                int tid = omp_get_thread_num();
                HeuCtx ctx(cur->state, rowBase);
                FastHashSet<Sig, SigHash> localSeen; localSeen.reserve(beamWidth * 32);

#pragma omp for schedule(static)
                for (int rid = 0; rid < (int)rotMaps.size(); ++rid) {
                    const RotMap& Mref = rotMaps[rid];

                    if (!(Mref.x >= cur->leftBound && Mref.y >= cur->topBound &&
                          (Mref.x + Mref.s) <= cur->rightBound && (Mref.y + Mref.s) <= cur->bottomBound)) continue;

                    if (isRotationNoop(cur->state, Mref)) continue;
                    int nearScore = ctx.nonadj_in_square(Mref);
                    if (Mref.s >= LARGE_S_SKIP_THRESHOLD && nearScore == 0) continue;

                    vector<int> affected; affected.reserve(32);
                    ctx.mark_once(cur->state.cells.data(), Mref, affected);
                    if (affected.empty()) continue;

                    int g = cur->g + 1;
                    int h_lb = ctx.eval_after_rotate_LB_with_list(Mref, affected);

                    int prSmall = (Mref.s <= 4 ? 5 - Mref.s : 0);
                    int prNear = std::min(8, nearScore);
                    int f_lb = g + h_lb - (prSmall + prNear);

                    int cut = std::min(localTop[tid].worst(), aspCutGlobal);
                    if (f_lb >= cut) continue;

                    Sig childSig = sigAfterRotateNoApply(cur->state, cur->sig, Mref);
                    if (visited.find(childSig) != visited.end()) continue;
                    if (!localSeen.insert(childSig).second) continue;

                    int h = ctx.eval_after_rotate_full_with_list(Mref, affected);
                    int f = g + h;
                    if (f < localTop[tid].worst()) {
                        Cand c; c.g = g; c.h = h; c.f = f; c.rx = Mref.x; c.ry = Mref.y; c.rs = Mref.s; c.parent = cur; c.map = &Mref; c.sig = childSig;
                        localTop[tid].consider(std::move(c));
                    }
                }
            }
        } else
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
                FastHashSet<Sig, SigHash> localSeen; localSeen.reserve(beamWidth * 16);

                for (const RotMap& Mref : rotMaps) {
                    if (!(Mref.x >= cur->leftBound && Mref.y >= cur->topBound &&
                          (Mref.x + Mref.s) <= cur->rightBound && (Mref.y + Mref.s) <= cur->bottomBound)) continue;

                    if (isRotationNoop(cur->state, Mref)) continue;
                    int nearScore = ctx.nonadj_in_square(Mref);
                    if (Mref.s >= LARGE_S_SKIP_THRESHOLD && nearScore == 0) continue;

                    vector<int> affected; affected.reserve(32);
                    ctx.mark_once(cur->state.cells.data(), Mref, affected);
                    if (affected.empty()) continue;

                    int g = cur->g + 1;
                    int h_lb = ctx.eval_after_rotate_LB_with_list(Mref, affected);

                    int prSmall = (Mref.s <= 4 ? 5 - Mref.s : 0);
                    int prNear = std::min(8, nearScore);
                    int f_lb = g + h_lb - (prSmall + prNear);

                    int cut = std::min(localTop[tid].worst(), aspCutGlobal);
                    if (f_lb >= cut) continue;

                    Sig childSig = sigAfterRotateNoApply(cur->state, cur->sig, Mref);
                    if (visited.find(childSig) != visited.end()) continue;
                    if (!localSeen.insert(childSig).second) continue;

                    int h = ctx.eval_after_rotate_full_with_list(Mref, affected);
                    int f = g + h;
                    if (f < localTop[tid].worst()) {
                        Cand c; c.g = g; c.h = h; c.f = f; c.rx = Mref.x; c.ry = Mref.y; c.rs = Mref.s; c.parent = cur; c.map = &Mref; c.sig = childSig;
                        localTop[tid].consider(std::move(c));
                    }
                }
            }
        }

        // 収集
        vector<Cand> pool; size_t est = 0; for (auto& lt : localTop) est += min(lt.K, (int)lt.heap.size());
        pool.reserve(est); for (auto& lt : localTop) lt.drain_to(pool);
        if (pool.empty()) break;

        if (pool.size() > (size_t)beamWidth) {
            std::nth_element(pool.begin(), pool.begin() + beamWidth, pool.end(), by_f);
            pool.resize(beamWidth);
        }
        std::sort(pool.begin(), pool.end(), by_f);

        // 同層内の署名重複除去
        FastHashSet<Sig, SigHash> nextSigSet; nextSigSet.reserve(pool.size() * 2 + 3);

        // 子ノード生成
        vector<shared_ptr<Node_Beam>> nextBeam; nextBeam.reserve(pool.size());
        for (const auto& c : pool) {
            if (!nextSigSet.insert(c.sig).second) continue;

            auto child = make_shared<Node_Beam>(fieldSize);
            child->state = c.parent->state;
            rotateCW_inplace(child->state, *c.map);
            child->sig = c.sig;
            child->g = c.g; child->h = c.h;
            child->rx = c.rx; child->ry = c.ry; child->rs = c.rs;
            child->parent = c.parent;

            child->leftBound   = c.parent->leftBound;
            child->topBound    = c.parent->topBound;
            child->rightBound  = c.parent->rightBound;
            child->bottomBound = c.parent->bottomBound;
            applyFixationLoop(child->state, child->leftBound, child->topBound, child->rightBound, child->bottomBound);

            nextBeam.push_back(std::move(child));
        }

        if (nextBeam.empty()) break;

        // h最小の更新
        auto itH = std::min_element(nextBeam.begin(), nextBeam.end(),
            [](const auto& a, const auto& b) { return a->h < b->h; });
        int curBestH = (*itH)->h;
        if (curBestH < bestH) { bestH = curBestH; noImp = 0; bestNodeByH = *itH; }
        else { if (++noImp >= maxNoImp) break; }

        // ペア数最大（同数なら手数が短い方）
        for (const auto& ch : nextBeam) {
            int pairs = countAdjacentPairs(ch->state);
            if (pairs > bestPairs || (pairs == bestPairs && ch->g < bestNodeByPairs->g)) {
                bestPairs = pairs; bestNodeByPairs = ch;
            }
        }

        beam = std::move(nextBeam);
    }

    // 経路復元（ペア数最大の経路を採用）
    solutionSteps.clear();
    auto p = bestNodeByPairs;
    while (p->parent) { solutionSteps.emplace_back(p->rx, p->ry, p->rs); p = p->parent; }
    std::reverse(solutionSteps.begin(), solutionSteps.end());

    // 同一領域の連続回転を mod4 で簡約
    compressSolutionSteps();

    return bestNodeByPairs;
}

// 回転列の簡約
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
        int keep = run % 4; // 4回は恒等
        for (int k = 0; k < keep; ++k) out.emplace_back(x, y, s);
        i = j;
    }
    solutionSteps.swap(out);
}

// 出力
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

void App::run() {
    auto finalNode = beamSearch(field, 3000);

    if (solutionSteps.empty()) {
        std::cout << "解が見つかりませんでした。\n";
    } else {
        printSolution_write();
        Field finalField = finalNode->state;
        writeSummaryJson(finalField);
        std::cout << "result.json " << solutionSteps.size() << " \n";

        const char post[] =
            "curl -X POST http://10.30.6.194:3000/submit -F \"result=@result.json\" -F \"numnum=@summary.json\"";
        cmd(post);
    }
}

int main() {
    const char get[] = "curl http://10.30.6.194:3000/get_json -o match.json";
    App::cmd(get);

    try {
        App app("match.json");
        app.run();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return -1;
    }
    return 0;
}
