#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

static std::string quote(const fs::path& p) {
#if defined(_WIN32) && !defined(__CYGWIN__)
    return "\"" + p.string() + "\"";
#else
    std::string out = "'";
    for (char ch : p.string()) out += (ch == '\'' ? "'\\''" : std::string(1, ch));
    out += "'";
    return out;
#endif
}

int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "Usage: " << argv[0]
                  << " <train.dat> <sigma> <rules.dat> <global|cns> <p|pn> <min_netconf>\n";
        return EXIT_FAILURE;
    }
    const std::string support = argv[4];
    const std::string evidence = argv[5];
    if (support != "global" && support != "cns") {
        std::cerr << "Error: support mode must be global or cns.\n";
        return EXIT_FAILURE;
    }
    if (evidence != "p" && evidence != "pn") {
        std::cerr << "Error: evidence mode must be p or pn.\n";
        return EXIT_FAILURE;
    }
    double min_netconf = 0.0;
    try { min_netconf = std::stod(argv[6]); }
    catch (...) { std::cerr << "Error: invalid min_netconf.\n"; return EXIT_FAILURE; }
    if (min_netconf < 0.0) {
        std::cerr << "Error: min_netconf must be >= 0.0.\n";
        return EXIT_FAILURE;
    }

    fs::path self = fs::absolute(argv[0]);
    fs::path dir = self.parent_path();
#if defined(_WIN32) || defined(__CYGWIN__)
    const char* ext = ".exe";
#else
    const char* ext = "";
#endif
    const std::string backend_name = "miner_" + support + "_" + evidence + ext;
    const fs::path backend = dir / backend_name;
    if (!fs::exists(backend)) {
        std::cerr << "Error: backend not found: " << backend << "\n";
        return EXIT_FAILURE;
    }

    std::cerr << "DMC dispatcher: support=" << support
              << " evidence=" << evidence
              << " sigma=" << argv[2]
              << " min_netconf=" << argv[6]
              << " backend=" << backend.filename().string() << "\n";

    std::ostringstream cmd;
    cmd << quote(backend) << ' ' << quote(argv[1]) << ' ' << argv[2] << ' '
        << quote(argv[3]) << ' ' << argv[6];
    const int rc = std::system(cmd.str().c_str());
    if (rc != 0) {
        std::cerr << "Error: backend exited with status " << rc << "\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
