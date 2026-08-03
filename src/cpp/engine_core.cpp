// engine_core.cpp
#include "engine_core.h"

#include "server-common.h"
#include "server-schema.h"
#include "common.h"
#include "llama.h"

#include <stdexcept>
#include <utility>
#include <vector>

using json = nlohmann::ordered_json;

namespace {

// Maps llama-server's stop_type string (from server-task.cpp
// stop_type_to_str(): "eos"/"word"/"limit"/"none") onto the two
// OpenAI-style finish_reason values this library exposes. "eos"/"word" mean
// the model stopped itself (matches oaicompat's own to_json_oaicompat()
// mapping); "limit" means max_tokens was hit. Any other/unrecognized value
// (including "none", which should not occur for a genuinely-finished
// result) defaults to "stop" rather than an empty string, so the Python-side
// Literal["stop", "length"] contract is never violated.
std::string map_finish_reason(const std::string & stop_type_str) {
    if (stop_type_str == "limit") return "length";
    return "stop";
}

EngineCore::Timings extract_timings(const json & j) {
    EngineCore::Timings t;
    if (!j.contains("timings")) return t;
    const auto & tj = j.at("timings");
    t.prompt_n             = tj.value("prompt_n", 0);
    t.prompt_ms            = tj.value("prompt_ms", 0.0);
    t.prompt_per_second    = tj.value("prompt_per_second", 0.0);
    t.predicted_n          = tj.value("predicted_n", 0);
    t.predicted_ms         = tj.value("predicted_ms", 0.0);
    t.predicted_per_second = tj.value("predicted_per_second", 0.0);
    return t;
}

} // namespace

EngineCore::EngineCore(const std::string & model_path, const std::string & mmproj_path,
                        int n_ctx, int n_gpu_layers, int n_parallel) {
    params_.model.path         = model_path;
    params_.mmproj.path        = mmproj_path;
    params_.n_gpu_layers       = n_gpu_layers;
    params_.mmproj_use_gpu     = (n_gpu_layers > 0);
    params_.n_parallel         = n_parallel;
    params_.n_ctx              = n_ctx;
    params_.cont_batching      = true;
    params_.special            = true;
    params_.sleep_idle_seconds = -1;  // never sleep (else model gets unloaded mid-use)
    params_.chat_template      = "";  // use the model's own built-in template
    params_.use_jinja          = true;

    llama_backend_init();
    llama_numa_init(params_.numa);

    if (!ctx_.load_model(params_)) {
        llama_backend_free();
        throw std::runtime_error("load_model failed");
    }
    meta_ = std::make_unique<server_context_meta>(ctx_.get_meta());
    loop_thread_ = std::thread([this] { ctx_.start_loop(); });
    started_ = true;
}

EngineCore::~EngineCore() {
    if (started_) {
        ctx_.terminate();
        if (loop_thread_.joinable()) loop_thread_.join();
    }
    llama_backend_free();
}
