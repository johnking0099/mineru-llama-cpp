// engine_core.cpp
#include "engine_core.h"

#include "server-common.h"
#include "server-schema.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "gguf.h"

#include <stdexcept>
#include <string>
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

// Reads the model's training context length ({arch}.context_length) straight
// from the GGUF metadata, without loading weights. Needed because we default
// n_ctx_seq to n_ctx_train, but n_ctx (and thus the KV cache) must be sized
// *before* the model finishes loading -- at which point llama.cpp's own
// llama_model_n_ctx_train() isn't available yet. Metadata-only parsing
// (no_alloc, ctx=NULL) is cheap. Returns 0 if the file/key can't be read, so
// the caller can fall back to llama.cpp's own default handling.
int read_n_ctx_train_from_gguf(const std::string & model_path) {
    gguf_init_params gp{ /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context * ctx = gguf_init_from_file(model_path.c_str(), gp);
    if (ctx == nullptr) return 0;

    int result = 0;
    const int64_t arch_id = gguf_find_key(ctx, "general.architecture");
    if (arch_id >= 0 && gguf_get_kv_type(ctx, arch_id) == GGUF_TYPE_STRING) {
        const std::string key = std::string(gguf_get_val_str(ctx, arch_id)) + ".context_length";
        const int64_t ctx_id = gguf_find_key(ctx, key.c_str());
        if (ctx_id >= 0 && gguf_get_kv_type(ctx, ctx_id) == GGUF_TYPE_UINT32) {
            result = (int) gguf_get_val_u32(ctx, ctx_id);
        }
    }
    gguf_free(ctx);
    return result;
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
                        int n_ctx_seq, int n_gpu_layers, int n_parallel,
                        int32_t verbosity, int32_t n_threads) {
    params_.model.path          = model_path;
    params_.mmproj.path         = mmproj_path;
    params_.n_gpu_layers        = n_gpu_layers;
    params_.mmproj_use_gpu      = (n_gpu_layers > 0);
    params_.n_parallel          = n_parallel;

    // We expose per-slot context (n_ctx_seq), not llama.cpp's total n_ctx.
    // n_ctx_seq == 0 means "the model's training context length"; read it
    // straight from GGUF metadata since llama_model_n_ctx_train() isn't
    // available until after the model loads (which is too late -- the KV
    // cache is sized during load). If the read fails, fall back to letting
    // llama.cpp default n_ctx itself (params_.n_ctx = 0).
    if (n_ctx_seq == 0) {
        n_ctx_seq = read_n_ctx_train_from_gguf(model_path);
    }
    params_.n_ctx = (n_ctx_seq > 0) ? n_ctx_seq * n_parallel : 0;

    // NOT unified: with kv_unified = false the KV cache is hard-partitioned
    // into n_parallel private streams of n_ctx / n_parallel cells each. Since
    // we size n_ctx = n_ctx_seq * n_parallel above, each slot gets exactly
    // n_ctx_seq cells that no other slot can touch. This makes the
    // "failed to find a memory slot" failure (a slot starved because its
    // peers hold the shared pool) structurally impossible: a request either
    // fits in its own n_ctx_seq (runs, or cleanly stops with finish_reason=
    // "length" at the cap) or is rejected up front if its prompt alone
    // exceeds n_ctx_seq -- never a load-dependent mid-batch failure. Total KV
    // memory is identical to the unified case (n_ctx cells either way); the
    // only thing that changes is partitioned vs shared. For an offline batch
    // engine that values predictable stability over letting one request
    // borrow idle peers' capacity, the hard partition is the right trade.
    params_.kv_unified          = false;
    params_.cont_batching       = true;
    
    // Engine requests disable prompt reuse, so retaining idle slots only
    // adds RAM-cache eviction work during offline multi-modal batches.
    params_.cache_idle_slots    = false;
    params_.cache_ram_mib       = 0;

    params_.special             = true;
    params_.sleep_idle_seconds  = -1;  // never sleep (else model gets unloaded mid-use)
    params_.chat_template       = "";  // use the model's own built-in template
    params_.use_jinja           = true;
    params_.verbosity           = verbosity;
    params_.cpuparams.n_threads = n_threads;

    // common_init() wires llama's own log callback (llama_log_set) through
    // to common_log -- without it, LLAMA_LOG_DEBUG lines from llama.cpp
    // internals (e.g. llama_context::set_embeddings) bypass the verbosity
    // threshold entirely and print unconditionally, same as every llama.cpp
    // CLI tool calls this before touching the backend (see e.g.
    // tools/server/server.cpp).
    common_init();
    common_log_set_verbosity_thold(params_.verbosity);

    // CLI entry points funnel cpuparams.n_threads through this before it
    // ever reaches llama_context -- skipping it means -1 ("auto") is never
    // resolved and silently falls back to GGML_DEFAULT_N_THREADS (4) deep
    // inside ggml-cpu.c, regardless of how many cores are actually
    // available.
    postprocess_cpu_params(params_.cpuparams, nullptr);

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
