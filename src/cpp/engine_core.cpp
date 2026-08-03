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

EngineCore::GenerateResult EngineCore::generate(const std::string & body_json) {
    json body = json::parse(body_json);  // throws nlohmann::json::parse_error on malformed input;
                                          // propagates uncaught to the binding layer (design spec §4.2 point 3)

    server_response_reader rd = ctx_.get_response_reader();
    server_task task(SERVER_TASK_TYPE_COMPLETION);

    {
        std::lock_guard<std::mutex> lk(parse_mu_);
        task.id = rd.get_new_id();

        std::vector<raw_buffer> files;
        // Reuses llama-server's own OpenAI-compatible chat parsing ("Path A"):
        // renders the jinja chat template, extracts image_url data URIs into
        // files, and normalizes sampling params into `parsed`. Throws
        // std::runtime_error/std::invalid_argument on bad image_url etc.
        json parsed = oaicompat_chat_params_parse(body, meta_->chat_params, files);
        const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx_.get_llama_context()));
        task.params = server_schema::eval_llama_cmpl_schema(
            vocab, params_, meta_->slot_n_ctx, meta_->logit_bias_eog, parsed);
        task.params.stream       = false;
        task.params.res_type     = TASK_RESPONSE_TYPE_NONE;
        task.params.cache_prompt = false;  // don't reuse KV across requests (fixes Tier2 #3's "empty prompt" bug)
        task.cli        = true;
        task.cli_prompt = parsed.at("prompt").get<std::string>();
        task.cli_files  = std::move(files);
    }

    rd.post_task(std::move(task));

    GenerateResult result;
    auto r = rd.next([]() { return false; });  // blocks until the result arrives; no GIL concept at this layer
    if (!r) {
        result.is_error   = true;
        result.error_json = R"json({"type":"server_error","message":"no result (stopped)"})json";
        return result;
    }
    if (r->is_error()) {
        result.is_error   = true;
        result.error_json = r->to_json().dump();
        return result;
    }

    json j = r->to_json();
    result.content         = j.value("content", std::string());
    result.finish_reason    = map_finish_reason(j.value("stop_type", std::string()));
    result.tokens_evaluated = j.value("tokens_evaluated", 0);
    result.tokens_predicted = j.value("tokens_predicted", 0);
    result.timings          = extract_timings(j);
    return result;
}

EngineCore::StreamHandle::StreamHandle(std::unique_ptr<server_response_reader> rd)
    : rd_(std::move(rd)) {}

EngineCore::Chunk EngineCore::StreamHandle::next_chunk() {
    Chunk chunk;
    if (done_) {
        // Caller kept calling after the final chunk; report as an immediate
        // final (empty-delta) chunk rather than blocking on a reader that
        // has nothing left to give.
        chunk.is_final = true;
        chunk.finish_reason = "stop";
        return chunk;
    }

    auto r = rd_->next([]() { return false; });
    if (!r) {
        chunk.is_error   = true;
        chunk.error_json = R"json({"type":"server_error","message":"no result (stopped)"})json";
        done_ = true;
        return chunk;
    }
    if (r->is_error()) {
        chunk.is_error   = true;
        chunk.error_json = r->to_json().dump();
        done_ = true;
        return chunk;
    }

    // The first partial chunk (is_begin, an SSE-header signal with no
    // content) serializes to a null json, not an object — see server-task.cpp
    // server_task_result_cmpl_partial::to_json(). Treat it as an empty delta.
    json j = r->to_json();
    if (!j.is_null()) {
        chunk.delta = j.value("content", std::string());
    }

    if (r->is_stop()) {
        chunk.is_final          = true;
        chunk.finish_reason     = map_finish_reason(j.is_null() ? std::string() : j.value("stop_type", std::string()));
        chunk.tokens_evaluated  = j.is_null() ? 0 : j.value("tokens_evaluated", 0);
        chunk.tokens_predicted  = j.is_null() ? 0 : j.value("tokens_predicted", 0);
        chunk.timings           = j.is_null() ? Timings{} : extract_timings(j);
        done_ = true;
    }
    return chunk;
}

EngineCore::StreamHandle EngineCore::generate_stream(const std::string & body_json) {
    json body = json::parse(body_json);  // throws on malformed input; propagates to binding layer

    // IMPORTANT: construct the ONE server_response_reader that will live for
    // this stream's entire lifetime, then mutate it in place through the
    // pointer (get_new_id(), post_task()) — never copy or move the
    // underlying object again after this line.
    //
    // Note on the line below: std::make_unique<T>(get_response_reader())
    // does NOT elide a copy here — make_unique takes its argument through a
    // forwarding-reference template parameter, which is not a
    // guaranteed-copy-elision context, so the compiler genuinely invokes
    // server_response_reader's implicit copy constructor once (confirmed via
    // -Wdeprecated-copy-with-user-provided-dtor). This is harmless ONLY
    // because the reader returned by get_response_reader() is still empty
    // (no task posted, id_tasks empty) at this exact point, so the copy's
    // destructor-triggered stop() is a no-op. The invariant we must actually
    // preserve is: never touch a *populated* reader (one that has had
    // post_task() called on it) through anything other than the single
    // unique_ptr from here on — see the correctness note at the top of
    // Phase B.
    auto reader = std::make_unique<server_response_reader>(ctx_.get_response_reader());

    server_task task(SERVER_TASK_TYPE_COMPLETION);
    {
        std::lock_guard<std::mutex> lk(parse_mu_);
        task.id = reader->get_new_id();

        std::vector<raw_buffer> files;
        json parsed = oaicompat_chat_params_parse(body, meta_->chat_params, files);
        const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx_.get_llama_context()));
        task.params = server_schema::eval_llama_cmpl_schema(
            vocab, params_, meta_->slot_n_ctx, meta_->logit_bias_eog, parsed);
        task.params.stream       = true;
        task.params.res_type     = TASK_RESPONSE_TYPE_NONE;
        task.params.cache_prompt = false;
        task.cli        = true;
        task.cli_prompt = parsed.at("prompt").get<std::string>();
        task.cli_files  = std::move(files);
    }

    reader->post_task(std::move(task));
    return StreamHandle(std::move(reader));  // only the unique_ptr moves; the reader's identity never changes
}
