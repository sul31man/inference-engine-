#include "infer_engine/io/model_loader.hpp"
#include "infer_engine/model/config.hpp"
#include "infer_engine/model/weights.hpp"
#include "infer_engine/runtime/runtime_ctx.hpp"
#include "infer_engine/core/tensor.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <iomanip>
#include <unistd.h>

namespace iegen {

static int64_t parse_int_field(const std::string& content, const std::string& key, int64_t fallback = 0) {
    const std::string k = "\"" + key + "\"";
    size_t pos = content.find(k);
    if (pos == std::string::npos) return fallback;
    size_t colon = content.find(':', pos);
    if (colon == std::string::npos) return fallback;
    size_t start = colon + 1;
    while (start < content.size() && (content[start] == ' ' || content[start] == '\t')) start++;
    size_t end = start;
    while (end < content.size() && (std::isdigit(static_cast<unsigned char>(content[end])) || content[end] == '-')) end++;
    if (start == end) return fallback;
    try { return std::stoll(content.substr(start, end - start)); } catch (...) { return fallback; }
}

#if !defined(_WIN32)
static std::string run_python_with_stdin(const std::string& py, const std::string& arg, const std::string& stdin_text) {
    char in_name[] = "/tmp/tokenizer_in_XXXXXX";
    char out_name[] = "/tmp/tokenizer_out_XXXXXX";
    int in_fd = mkstemp(in_name);
    int out_fd = mkstemp(out_name);
    if (in_fd == -1 || out_fd == -1) throw std::runtime_error("mkstemp failed");
    {
        FILE* f = fdopen(in_fd, "wb");
        if (!f) throw std::runtime_error("fdopen input failed");
        fwrite(stdin_text.data(), 1, stdin_text.size(), f);
        fclose(f);
    }
    close(out_fd);
    std::ostringstream cmd;
    cmd << "python3 -c \"" << py << "\" " << arg
        << " < " << in_name << " > " << out_name << " 2>/dev/null";
    std::system(cmd.str().c_str());
    std::ifstream out(out_name, std::ios::binary);
    std::stringstream ss; ss << out.rdbuf(); out.close();
    std::remove(in_name); std::remove(out_name);
    return ss.str();
}
#endif

class Tokenizer {
public:
    explicit Tokenizer(const std::string& model_dir)
        : tokenizer_json_path_(model_dir + "/tokenizer.json") {}

    std::vector<int64_t> encode(const std::string& text) {
#if !defined(_WIN32)
        try {
            const std::string py =
                "import sys\n"
                "from tokenizers import Tokenizer\n"
                "tok = Tokenizer.from_file(sys.argv[1])\n"
                "text = sys.stdin.read()\n"
                "enc = tok.encode(text)\n"
                "print(' '.join(map(str, enc.ids)))\n";
            std::string out = run_python_with_stdin(py, "\"" + escape_quotes(tokenizer_json_path_) + "\"", text);
            std::vector<int64_t> ids; std::istringstream iss(out); std::string t;
            while (iss >> t) { try { ids.push_back(std::stoll(t)); } catch (...) {} }
            if (!ids.empty()) return ids;
        } catch (...) {}
#endif
        std::vector<int64_t> ids; ids.reserve(text.size());
        for (unsigned char c : text) ids.push_back(static_cast<int64_t>(c));
        return ids;
    }

    std::string decode(const std::vector<int64_t>& ids) {
#if !defined(_WIN32)
        try {
            const std::string py =
                "import sys\n"
                "from tokenizers import Tokenizer\n"
                "tok = Tokenizer.from_file(sys.argv[1])\n"
                "ids = list(map(int, sys.stdin.read().split()))\n"
                "print(tok.decode(ids))\n";
            std::ostringstream ids_in;
            for (size_t i = 0; i < ids.size(); ++i) { if (i) ids_in << ' '; ids_in << ids[i]; }
            std::string out = run_python_with_stdin(py, "\"" + escape_quotes(tokenizer_json_path_) + "\"", ids_in.str());
            if (!out.empty()) return out;
        } catch (...) {}
#endif
        std::string s; s.reserve(ids.size());
        for (int64_t id : ids) s.push_back(static_cast<char>(static_cast<unsigned char>(id)));
        return s;
    }

private:
    static std::string escape_quotes(const std::string& s) {
        std::string out; out.reserve(s.size());
        for (char c : s) { if (c == '"') out.push_back('\\'); out.push_back(c); }
        return out;
    }
    std::string tokenizer_json_path_;
};

class Model {
public:
    explicit Model(const std::string& model_dir, int64_t max_seq_len = 2048) {
        {
            std::ifstream f(model_dir + "/config.json");
            if (f) {
                std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
                eos_token_id_ = parse_int_field(content, "eos_token_id", 2);
            }
        }
        ie::load_mistral_safetensors(model_dir, cfg_, weights_);
        ctx_ = std::make_unique<ie::RuntimeCtx>(cfg_, weights_, max_seq_len);
        const int64_t head_dim = cfg_.d_model / cfg_.n_heads;
        const double kv_gb = 2.0 * static_cast<double>(cfg_.n_layers) * static_cast<double>(max_seq_len)
            * static_cast<double>(cfg_.n_kv_heads) * static_cast<double>(head_dim) * 2.0
            / (1024.0 * 1024.0 * 1024.0);
        std::cout << "[Config] layers=" << cfg_.n_layers
                  << " n_heads=" << cfg_.n_heads
                  << " n_kv_heads=" << cfg_.n_kv_heads
                  << " head_dim=" << head_dim
                  << " max_seq_len=" << max_seq_len
                  << " kv_dtype=F16 expected_kv_gb~" << std::fixed << std::setprecision(2) << kv_gb << "\n";
    }

    std::vector<float> forward_tokens(const std::vector<int64_t>& token_ids) {
        if (token_ids.empty()) return {};
        std::vector<float> last_logits;
        for (int64_t pos = 0; pos < static_cast<int64_t>(token_ids.size()); ++pos) {
            const int64_t tok = token_ids[static_cast<size_t>(pos)];
            ie::Tensor logits = ctx_->forward_decode(static_cast<int32_t>(tok), pos);
            const float* lp = logits.view.ptr<const float>();
            last_logits.assign(lp, lp + cfg_.vocab_size);
        }
        return last_logits;
    }

    int64_t eos_token_id() const { return eos_token_id_; }
    const ie::ModelCfg& cfg() const { return cfg_; }

private:
    ie::ModelCfg cfg_{};
    ie::ModelWeights weights_{};
    std::unique_ptr<ie::RuntimeCtx> ctx_{};
    int64_t eos_token_id_{2};
};

static int64_t argmax_id(const std::vector<float>& logits) {
    if (logits.empty()) return 0;
    size_t best = 0; float bestv = logits[0];
    for (size_t i = 1; i < logits.size(); ++i) {
        if (logits[i] > bestv) { bestv = logits[i]; best = i; }
    }
    return static_cast<int64_t>(best);
}

} // namespace iegen

int main(int argc, char** argv) {
    using namespace iegen;
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_dir> [--max_new_tokens N] [--prompt \"text...\"]\n";
        return 1;
    }

    std::string model_dir = argv[1];
    int max_new_tokens = 50;
    int64_t max_seq_len = 2048;
    std::string prompt;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--max_new_tokens" && i + 1 < argc) {
            max_new_tokens = std::atoi(argv[++i]);
        } else if (a == "--max_seq_len" && i + 1 < argc) {
            max_seq_len = std::atoll(argv[++i]);
        } else if (a == "--prompt" && i + 1 < argc) {
            prompt = argv[++i];
        }
    }
    if (prompt.empty()) {
        std::ostringstream ss; ss << std::cin.rdbuf(); prompt = ss.str();
    }

    try {
        std::cout << "Loading tokenizer...\n";
        Tokenizer tokenizer(model_dir);
        std::cout << "Encoding prompt...\n";
        std::vector<int64_t> input_ids = tokenizer.encode(prompt);
        if (input_ids.empty()) { std::cerr << "Empty encoded prompt.\n"; return 1; }

        std::cout << "Loading model...\n";
        Model model(model_dir, max_seq_len);

        std::cout << "Prefill on " << input_ids.size() << " tokens...\n";
        std::vector<float> logits = model.forward_tokens(input_ids);

        const int64_t eos_id = model.eos_token_id();
        std::cout << "Generating up to " << max_new_tokens << " tokens (eos=" << eos_id << ")...\n";
        for (int step = 0; step < max_new_tokens; ++step) {
            int64_t next_id = argmax_id(logits);
            input_ids.push_back(next_id);
            if (next_id == eos_id) { std::cout << "EOS at step " << step << "\n"; break; }
            logits = model.forward_tokens({next_id});
        }

        std::cout << "Decoding...\n";
        std::string out = tokenizer.decode(input_ids);
        std::cout << "\n=== OUTPUT ===\n" << out << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}


