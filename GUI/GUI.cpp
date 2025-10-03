#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <cstdio>   // snprintf
#include "json.hpp"

using json = nlohmann::json;
using namespace std;

struct Operation {
    int x{ 0 }, y{ 0 }, n{ 0 }; // 左上座標 (x,y), 正方回転サイズ n
};

struct Field {
    int N{ 0 };
    std::vector<int> cells; // row-major
    Field() = default;
    explicit Field(int n) : N(n), cells(n* n, 0) {}
    inline int& at(int y, int x) { return cells[y * N + x]; }
    inline int  at(int y, int x) const { return cells[y * N + x]; }
};

// ---- n別 累計プレフィックス ----
struct CumStats {
    std::vector<int> uniqueNs;                 // 昇順ユニーク n
    std::vector<std::vector<int>> pref;        // pref[step][k] = step手後の n=uniqueNs[k] の回数
};

CumStats build_cum_stats(const std::vector<Operation>& ops) {
    CumStats st;
    // unique Ns
    {
        std::vector<int> ns; ns.reserve(ops.size());
        for (auto& o : ops) if (o.n > 0) ns.push_back(o.n);
        std::sort(ns.begin(), ns.end());
        ns.erase(std::unique(ns.begin(), ns.end()), ns.end());
        st.uniqueNs = std::move(ns);
    }
    const int K = (int)st.uniqueNs.size();
    std::unordered_map<int, int> idx; idx.reserve(K * 2 + 1);
    for (int i = 0; i < K; ++i) idx[st.uniqueNs[i]] = i;

    st.pref.assign(ops.size() + 1, std::vector<int>(K, 0));
    for (size_t i = 0; i < ops.size(); ++i) {
        st.pref[i + 1] = st.pref[i];
        auto it = idx.find(ops[i].n);
        if (it != idx.end()) st.pref[i + 1][it->second] += 1;
    }
    return st;
}

// --------- JSON I/O ---------
bool load_match_json(const std::string& path, Field& outField) {
    ifstream ifs(path);
    if (!ifs) { cerr << "[Error] cannot open file: " << path << endl; return false; }
    json j;
    try { ifs >> j; }
    catch (const std::exception& e) { cerr << "[Error] JSON parse error: " << e.what() << endl; return false; }

    try {
        int N = j["problem"]["field"]["size"].get<int>();
        const auto& ents = j["problem"]["field"]["entities"];
        if ((int)ents.size() != N) { cerr << "[Error] entities row count mismatch with size.\n"; return false; }
        Field f(N);
        for (int y = 0; y < N; ++y) {
            if ((int)ents[y].size() != N) { cerr << "[Error] entities col count mismatch on row " << y << ".\n"; return false; }
            for (int x = 0; x < N; ++x) f.at(y, x) = ents[y][x].get<int>();
        }
        outField = std::move(f);
        return true;
    }
    catch (const std::exception& e) {
        cerr << "[Error] match.json content error: " << e.what() << endl; return false;
    }
}

bool load_result_json(const std::string& path, std::vector<Operation>& ops) {
    ifstream ifs(path);
    if (!ifs) { cerr << "[Error] cannot open file: " << path << endl; return false; }
    json j;
    try { ifs >> j; }
    catch (const std::exception& e) { cerr << "[Error] JSON parse error: " << e.what() << endl; return false; }

    try {
        if (!j.contains("ops") || !j["ops"].is_array()) { cerr << "[Error] result.json does not have array 'ops'.\n"; return false; }
        ops.clear();
        for (const auto& o : j["ops"]) {
            Operation op;
            op.x = o.at("x").get<int>();
            op.y = o.at("y").get<int>();
            op.n = o.at("n").get<int>();
            ops.push_back(op);
        }
        return true;
    }
    catch (const std::exception& e) {
        cerr << "[Error] result.json content error: " << e.what() << endl; return false;
    }
}

// --------- 回転（時計回り / 反時計回り） ---------
void rotate_submatrix_cw(Field& f, int x, int y, int s) {
    if (s <= 0 || x < 0 || y < 0 || x + s > f.N || y + s > f.N) return;
    std::vector<int> tmp(s * s);
    for (int dy = 0; dy < s; ++dy) for (int dx = 0; dx < s; ++dx) tmp[dy * s + dx] = f.at(y + dy, x + dx);
    for (int dy = 0; dy < s; ++dy) for (int dx = 0; dx < s; ++dx) {
        int ny = y + dx, nx = x + (s - 1 - dy);
        f.at(ny, nx) = tmp[dy * s + dx];
    }
}

void rotate_submatrix_ccw(Field& f, int x, int y, int s) {
    if (s <= 0 || x < 0 || y < 0 || x + s > f.N || y + s > f.N) return;
    std::vector<int> tmp(s * s);
    for (int dy = 0; dy < s; ++dy) for (int dx = 0; dx < s; ++dx) tmp[dy * s + dx] = f.at(y + dy, x + dx);
    for (int dy = 0; dy < s; ++dy) for (int dx = 0; dx < s; ++dx) {
        int ny = y + (s - 1 - dx), nx = x + dy;
        f.at(ny, nx) = tmp[dy * s + dx];
    }
}

// --------- 現在のペア数（上下左右で値が等しい隣接辺の数） ---------
int count_adjacent_pairs(const Field& f) {
    int N = f.N;
    int cnt = 0;
    // 横方向
    for (int y = 0; y < N; ++y) {
        for (int x = 0; x + 1 < N; ++x) {
            if (f.at(y, x) == f.at(y, x + 1)) ++cnt;
        }
    }
    // 縦方向
    for (int y = 0; y + 1 < N; ++y) {
        for (int x = 0; x < N; ++x) {
            if (f.at(y, x) == f.at(y + 1, x)) ++cnt;
        }
    }
    return cnt;
}

// --------- マンハッタン距離用：初期ID盤面の構築と総和計算 ---------
// 各セルに固有ID = (y*N + x) を持つ盤面を用意し、操作列と同様に回転させる。
// 現在位置 (y,x) にあるIDの「初期座標 (oy,ox)」との L1 距離を全セルで合計。
Field build_identity_field(int N) {
    Field idf(N);
    for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x)
            idf.at(y, x) = y * N + x;
    return idf;
}

long long total_manhattan_from_identity(const Field& idfield) {
    const int N = idfield.N;
    long long sum = 0;
    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            int id = idfield.at(y, x);
            int oy = id / N;
            int ox = id % N;
            sum += std::abs(y - oy) + std::abs(x - ox);
        }
    }
    return sum;
}

// --------- カラーマップ（値→色, 安定） ---------
static inline cv::Scalar hsv2bgr(float h_deg, float s, float v) {
    float C = v * s;
    float hh = fmodf(h_deg, 360.f) / 60.f; // [0,6)
    float X = C * (1.f - fabsf(fmodf(hh, 2.f) - 1.f));
    float r = 0, g = 0, b = 0;
    if (0.f <= hh && hh < 1.f) { r = C; g = X; b = 0; }
    else if (1.f <= hh && hh < 2.f) { r = X; g = C; b = 0; }
    else if (2.f <= hh && hh < 3.f) { r = 0; g = C; b = X; }
    else if (3.f <= hh && hh < 4.f) { r = 0; g = X; b = C; }
    else if (4.f <= hh && hh < 5.f) { r = X; g = 0; b = C; }
    else if (5.f <= hh && hh < 6.f) { r = C; g = 0; b = X; }
    float m = v - C;
    r += m; g += m; b += m;
    return cv::Scalar(b * 255.f, g * 255.f, r * 255.f); // BGR
}

struct Palette {
    std::unordered_map<int, cv::Scalar> color;
    cv::Scalar get(int v) {
        auto it = color.find(v);
        if (it != color.end()) return it->second;
        if (v < 0) { cv::Scalar c(60, 60, 60); color.emplace(v, c); return c; }
        float hue = fmodf(v * 137.508f, 360.f);
        float sat = 0.68f, val = 0.95f;
        cv::Scalar c = hsv2bgr(hue, sat, val);
        color.emplace(v, c);
        return c;
    }
};

// ---- 右パネル描画（n別 累計：一列表示 + 現在手数 + 現在ペア数 + 現在マンハッタン距離） ----
void draw_n_stats_panel_onecol(
    cv::Mat& img,
    const CumStats& st,
    int stepIndex,                 // 現在の手数（0..ops.size()）
    int pairCount,                 // 現在のペア数
    long long manhattanSum,        // 現在のマンハッタン距離（総和）
    int x0, int y0,                // 左上座標
    int panelW, int panelH
) {
    const int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double titleScale = 0.55;
    double lineScale = 0.48;

    // 背景
    cv::rectangle(img, cv::Rect(x0, y0, panelW, panelH), cv::Scalar(230, 230, 230), cv::FILLED);
    cv::rectangle(img, cv::Rect(x0, y0, panelW, panelH), cv::Scalar(200, 200, 200), 1);

    int y = y0 + 20;
    cv::putText(img, "n別 累計 (current/total)", cv::Point(x0 + 8, y), fontFace, titleScale, cv::Scalar(40, 40, 40), 2, cv::LINE_AA);
    cv::putText(img, "n別 累計 (current/total)", cv::Point(x0 + 8, y), fontFace, titleScale, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    y += 10;

    cv::line(img, cv::Point(x0 + 8, y + 6), cv::Point(x0 + panelW - 8, y + 6), cv::Scalar(210, 210, 210), 1);
    y += 22;

    const int K = (int)st.uniqueNs.size();
    const std::vector<int>& cur = st.pref[std::min((size_t)stepIndex, st.pref.size() - 1)];
    const std::vector<int>& tot = st.pref.back();

    // --- 現在の手数・総手数・現在のペア数・現在のマンハッタン距離（右パネル上部） ---
    int totalOps = 0; for (int v : tot) totalOps += v;
    {
        char b1[64], b2[64], b3[64], b4[80];
        snprintf(b1, sizeof(b1), "現在の手数: %d", stepIndex);
        snprintf(b2, sizeof(b2), "総手数    : %d", totalOps);
        snprintf(b3, sizeof(b3), "現在のペア: %d", pairCount);
        snprintf(b4, sizeof(b4), "現在のマンハッタン: %lld", manhattanSum);
        cv::putText(img, b1, cv::Point(x0 + 8, y), fontFace, lineScale, cv::Scalar(25, 25, 25), 2, cv::LINE_AA);
        cv::putText(img, b1, cv::Point(x0 + 8, y), fontFace, lineScale, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        y += 20;
        cv::putText(img, b2, cv::Point(x0 + 8, y), fontFace, lineScale, cv::Scalar(25, 25, 25), 2, cv::LINE_AA);
        cv::putText(img, b2, cv::Point(x0 + 8, y), fontFace, lineScale, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        y += 20;
        cv::putText(img, b3, cv::Point(x0 + 8, y), fontFace, lineScale, cv::Scalar(25, 25, 25), 2, cv::LINE_AA);
        cv::putText(img, b3, cv::Point(x0 + 8, y), fontFace, lineScale, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        y += 20;
        cv::putText(img, b4, cv::Point(x0 + 8, y), fontFace, lineScale, cv::Scalar(25, 25, 25), 2, cv::LINE_AA);
        cv::putText(img, b4, cv::Point(x0 + 8, y), fontFace, lineScale, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        y += 18;
        cv::line(img, cv::Point(x0 + 8, y + 6), cv::Point(x0 + panelW - 8, y + 6), cv::Scalar(210, 210, 210), 1);
        y += 18;
    }

    if (K == 0) {
        cv::putText(img, "No ops", cv::Point(x0 + 8, y + 16), fontFace, lineScale, cv::Scalar(30, 30, 30), 1, cv::LINE_AA);
        return;
    }

    for (int i = 0; i < K; ++i) {
        char buf[64];
        snprintf(buf, sizeof(buf), "n=%-2d : %d/%d", st.uniqueNs[i], cur[i], tot[i]);
        cv::putText(img, buf, cv::Point(x0 + 8, y), fontFace, lineScale, cv::Scalar(25, 25, 25), 2, cv::LINE_AA);
        cv::putText(img, buf, cv::Point(x0 + 8, y), fontFace, lineScale, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        y += 20;
    }
}

// --------- 描画（セルと文字を小さめ／右パネル一列付き） ---------
cv::Mat render_board(
    const Field& f,
    int stepIndex,                        // 現在「何手適用後」か（0..ops.size()）
    const std::vector<Operation>& ops,    // ハイライト用
    const CumStats* stats = nullptr,      // n別 累計
    const Field* idState = nullptr,       // ★マンハッタン距離用のID盤面（現在ステップのもの）
    int cellSize = -1,                    // 自動計算は -1
    int margin_left = 28,
    int margin_top = 82
) {
    int N = f.N;
    if (N <= 0) return cv::Mat();

    // セルサイズ自動調整（控えめ）
    if (cellSize <= 0) {
        int targetW = 800; // 以前より小さめ（1100→800）
        cellSize = std::max(10, std::min(56, targetW / std::max(1, N)));
    }

    // 右パネル（幅は文字幅から動的計算、一列）
    const int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double lineScale = 0.48;
    int panelW = 0, panelH = 0;

    // 右パネル寸法は n別統計がある時に確保（ない場合はパネル自体を表示しない）
    if (stats) {
        int K = (int)stats->uniqueNs.size();

        int maxTextW = 0, base = 0;
        if (K > 0) {
            const auto& cur = stats->pref[std::min((size_t)stepIndex, stats->pref.size() - 1)];
            const auto& tot = stats->pref.back();
            for (int i = 0; i < K; ++i) {
                char buf[64];
                snprintf(buf, sizeof(buf), "n=%-2d : %d/%d", stats->uniqueNs[i], cur[i], tot[i]);
                cv::Size ts = cv::getTextSize(buf, fontFace, lineScale, 1, &base);
                maxTextW = std::max(maxTextW, ts.width);
            }
        }
        // 上部の情報行（手数/ペア/マンハッタン）も考慮して少し広めに
        panelW = std::max(280, 8 + maxTextW + 16); // 以前 240
        // タイトル/罫線/上部情報(4行)/罫線/項目K行/余白
        panelH = 200 + K * 20;
    }

    int W = margin_left + N * cellSize + (stats ? (28 + panelW) : 0);
    int H = std::max(margin_top + N * cellSize + 36, panelH + 20);
    cv::Mat img(H, W, CV_8UC3, cv::Scalar(242, 242, 242));

    static Palette palette;

    // セルの塗り / 数字（小さめ）
    double fontScale = std::clamp(cellSize / 80.0, 0.35, 1.10);
    int thickness = 1;

    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            cv::Rect rc(margin_left + x * cellSize, margin_top + y * cellSize, cellSize, cellSize);
            int v = f.at(y, x);
            cv::Scalar col = palette.get(v);
            cv::rectangle(img, rc, col, cv::FILLED);

            // グリッド線（薄め）
            cv::rectangle(img, rc, cv::Scalar(210, 210, 210), 1);

            // 数字
            double lum = 0.2126 * (col[2] / 255.0) + 0.7152 * (col[1] / 255.0) + 0.0722 * (col[0] / 255.0);
            cv::Scalar txt = (lum < 0.5) ? cv::Scalar(240, 240, 240) : cv::Scalar(30, 30, 30);

            std::string s = std::to_string(v);
            int base = 0;
            cv::Size sz = cv::getTextSize(s, fontFace, fontScale, thickness, &base);
            cv::Point org(rc.x + (rc.width - sz.width) / 2, rc.y + (rc.height + sz.height) / 2 - 3);
            cv::putText(img, s, org, fontFace, fontScale, txt, thickness, cv::LINE_AA);
        }
    }

    // ハイライト: 直近の手
    if (stepIndex > 0 && stepIndex - 1 < (int)ops.size()) {
        const auto& mv = ops[stepIndex - 1];
        cv::Rect hi(margin_left + mv.x * cellSize, margin_top + mv.y * cellSize, mv.n * cellSize, mv.n * cellSize);
        cv::rectangle(img, hi, cv::Scalar(0, 0, 0), 2);
        cv::rectangle(img, hi, cv::Scalar(0, 230, 230), 2);

        // 回転方向のガイド
        std::string cw = "CW";
        double fs2 = std::clamp(cellSize / 36.0, 0.40, 1.00);
        int base2 = 0;
        cv::Size ts = cv::getTextSize(cw, fontFace, fs2, 1, &base2);
        cv::Point tg(hi.x + hi.width - ts.width - 5, hi.y + ts.height + 5);
        cv::putText(img, cw, tg, fontFace, fs2, cv::Scalar(20, 20, 20), 1, cv::LINE_AA);
        cv::putText(img, cw, tg, fontFace, fs2, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
    }

    // 現在のペア数・マンハッタン距離を計算
    int pairCount = count_adjacent_pairs(f);
    long long manhattanSum = 0;
    if (idState) {
        manhattanSum = total_manhattan_from_identity(*idState);
    }

    // タイトル / 情報
    cv::putText(img, "OpenCV Visualizer", cv::Point(12, 24), fontFace, 0.62, cv::Scalar(60, 60, 60), 2, cv::LINE_AA);
    cv::putText(img, "OpenCV Visualizer", cv::Point(12, 24), fontFace, 0.62, cv::Scalar(250, 250, 250), 1, cv::LINE_AA);

    // 情報行（現在の手数・現在のペア数・現在のマンハッタン距離）
    char buf[320];
    snprintf(buf, sizeof(buf),
        "Step: %d / %d  (current=%d)  Pairs: %d  Manh: %lld  (a: next, b: prev, q: quit)",
        stepIndex, (int)ops.size(), stepIndex, pairCount, manhattanSum);
    cv::putText(img, buf, cv::Point(12, 52), fontFace, 0.52, cv::Scalar(25, 25, 25), 2, cv::LINE_AA);
    cv::putText(img, buf, cv::Point(12, 52), fontFace, 0.52, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

    if (stepIndex > 0 && stepIndex - 1 < (int)ops.size()) {
        const auto& mv = ops[stepIndex - 1];
        char mbuf[128];
        snprintf(mbuf, sizeof(mbuf), "Last move: x=%d, y=%d, n=%d, rotate CW", mv.x, mv.y, mv.n);
        cv::putText(img, mbuf, cv::Point(12, 74), fontFace, 0.48, cv::Scalar(30, 30, 30), 2, cv::LINE_AA);
        cv::putText(img, mbuf, cv::Point(12, 74), fontFace, 0.48, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }
    else {
        const char* t = "Initial state";
        cv::putText(img, t, cv::Point(12, 74), fontFace, 0.48, cv::Scalar(30, 30, 30), 2, cv::LINE_AA);
        cv::putText(img, t, cv::Point(12, 74), fontFace, 0.48, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }

    // 右パネル（n別 累計：一列 + 現在手数 + 現在ペア数 + 現在マンハッタン）
    if (stats) {
        int x0 = margin_left + N * cellSize + 16;
        int y0 = 20;
        draw_n_stats_panel_onecol(img, *stats, stepIndex, pairCount, manhattanSum, x0, y0, panelW - 16, panelH);
    }

    return img;
}

// --------- 全ステート事前生成（戻るを高速化） ---------
std::vector<Field> materialize_states(const Field& start, const std::vector<Operation>& ops) {
    std::vector<Field> states;
    states.reserve(ops.size() + 1);
    states.push_back(start);
    Field cur = start;
    for (size_t i = 0; i < ops.size(); ++i) {
        const auto& mv = ops[i];
        if (mv.x < 0 || mv.y < 0 || mv.n <= 0 || mv.x + mv.n > cur.N || mv.y + mv.n > cur.N) {
            cerr << "[Warn] op[" << i << "] is out of range. (x=" << mv.x << ",y=" << mv.y << ",n=" << mv.n << ")\n";
        }
        else {
            rotate_submatrix_cw(cur, mv.x, mv.y, mv.n);
        }
        states.push_back(cur);
    }
    return states;
}

// --------- メインループ ---------
int main(int argc, char** argv) {
    ios::sync_with_stdio(false);

    std::string matchPath = "match.json";
    std::string resultPath = "result.json";
    if (argc >= 2) matchPath = argv[1];
    if (argc >= 3) resultPath = argv[2];

    Field start;
    if (!load_match_json(matchPath, start)) {
        cerr << "match.json の読み込みに失敗しました。" << endl;
        return 1;
    }

    std::vector<Operation> ops;
    if (!load_result_json(resultPath, ops)) {
        cerr << "result.json の読み込みに失敗しました。" << endl;
        return 1;
    }

    // n別 累計プレフィックスを構築
    CumStats stats = build_cum_stats(ops);

    // すべてのステップ状態を事前に作成（b で素早く戻れる）
    auto states = materialize_states(start, ops);

    // ★マンハッタン距離用：初期ID盤面を構築し、同じ操作列で事前に全ステップ分を構築
    Field idStart = build_identity_field(start.N);
    auto id_states = materialize_states(idStart, ops);

    int curStep = 0; // 0..ops.size()

    const std::string win = "Visualizer (a: next, b: prev, q: quit)";
    cv::namedWindow(win, cv::WINDOW_NORMAL);

    // 初回描画
    {
        cv::Mat frame = render_board(states[curStep], curStep, ops, &stats, &id_states[curStep]);
        if (!frame.empty()) {
            cv::imshow(win, frame);
            // ウィンドウサイズは控えめ
            cv::resizeWindow(win, std::min(1200, frame.cols), std::min(820, frame.rows));
        }
    }

    // 入力ループ
    while (true) {
        int key = cv::waitKeyEx(0); // 0: ブロッキング
        if (key == 'q' || key == 'Q' || key == 27 /*ESC*/) {
            break;
        }
        else if (key == 'a' || key == 'A' || key == 2555904 /*Right*/) {
            if (curStep < (int)ops.size()) ++curStep;
            cv::Mat frame = render_board(states[curStep], curStep, ops, &stats, &id_states[curStep]);
            cv::imshow(win, frame);
        }
        else if (key == 'b' || key == 'B' || key == 2424832 /*Left*/) {
            if (curStep > 0) --curStep;
            cv::Mat frame = render_board(states[curStep], curStep, ops, &stats, &id_states[curStep]);
            cv::imshow(win, frame);
        }
        else {
            // 他キーは無視して再描画
            cv::Mat frame = render_board(states[curStep], curStep, ops, &stats, &id_states[curStep]);
            cv::imshow(win, frame);
        }
    }

    return 0;
}
