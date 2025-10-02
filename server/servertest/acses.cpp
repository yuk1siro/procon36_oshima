#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <thread>
#include <chrono>
#include "json.hpp"

using json = nlohmann::json;

void cmd(const char cmd[]) {
    system(cmd);
}

int main() {
    const char get[] = "curl -v http://169.254.39.0:3000/ -H \"Procon-Token: player1\" -o match.json";
    const char post[] = "curl -i http://169.254.39.0:3000/ -X POST -H \"Procon-Token: player1\" -H\"Content-Type:application/json\" -d @result.json";

    cmd(get);

    std::string filename = "numnum.json";
    int countmax = 0;
    int stepsmin = 2000;

    while (true) {
        // ファイル読み込み

        std::ifstream ifs(filename, std::ios::in | std::ios::binary);
        if (!ifs) {
            std::cerr << "ファイルを開けません: " << filename << " (再試行します...)" << std::endl;
            continue;
        }

        std::ostringstream ss;
        ss << ifs.rdbuf();
        std::string content = ss.str();
        ifs.close();

        if (content.empty()) {
            std::cerr << "ファイルが空です。待機..." << std::endl;
            continue;
        }

        json j;
        try {
            j = json::parse(content);
        }
        catch (const json::parse_error& e) {
            std::cerr << "JSON 解析エラー: " << e.what() << " (再試行します)" << std::endl;
            continue;
        }

        if (!j.is_object()) {
            std::cerr << "JSON ルートがオブジェクトではありません。待機..." << std::endl;
            continue;
        }

        int count = j.at("count").get<int>();
        int steps = j.at("steps").get<int>();

        
        //大小関係
        if (count >= countmax && steps <= stepsmin) {
            std::cout << "read: count = " << count << ", steps = " << steps << std::endl;
            countmax = count + 1; stepsmin = steps - 1;
            std::cout << "(current best) count_max = " << countmax - 1 << ", steps_min = " << stepsmin + 1 << std::endl;
            cmd(post);
        }
        

    }
    
}
