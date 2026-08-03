// mineru_engine_spike.cpp — Tier1 #2 spike: prove the async/GIL bridge model.
//
// Goal: prove an Engine binding can (a) run llama-server's loop in a background
// thread, (b) release the GIL while blocking on a result, (c) be driven async
// via asyncio.run_in_executor so the event loop stays free during decode.
//
// This is NOT the full Engine API — just enough to de-risk the AsyncEngine
// threading model. Reuses Path A (oaicompat) for input parsing.

#include <pybind11/pybind11.h>

#include "server-context.h"
#include "server-queue.h"
#include "server-task.h"
#include "server-common.h"
#include "server-schema.h"
#include "common.h"
#include "llama.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

using json = nlohmann::ordered_json;
namespace py = pybind11;

class Engine {
public:
    server_context                       ctx;
    common_params                        params;
    std::unique_ptr<server_context_meta> meta;
    std::thread                          loop_thread;
    std::atomic<bool>                    started{false};
    // guard shared chat template rendering (Tier2 #3 will determine if needed)
    std::mutex                           parse_mu;

    Engine(const std::string & model, const std::string & mmproj, int n_ctx, int n_gpu_layers, int n_parallel) {
        params.model.path         = model;
        params.mmproj.path        = mmproj;
        params.n_gpu_layers       = n_gpu_layers;
        params.mmproj_use_gpu     = (n_gpu_layers > 0);  // allow isolating Metal
        params.n_parallel         = n_parallel;
        params.n_ctx              = n_ctx;
        params.cont_batching      = true;
        params.special            = true;
        params.sleep_idle_seconds = -1;  // never sleep (else model gets unloaded)
        params.chat_template      = "";
        params.use_jinja          = true;

        llama_backend_init();
        llama_numa_init(params.numa);

        if (!ctx.load_model(params)) throw std::runtime_error("load_model failed");
        meta = std::make_unique<server_context_meta>(ctx.get_meta());
        loop_thread = std::thread([this] { ctx.start_loop(); });
        started = true;
    }

    ~Engine() {
        if (started) {
            ctx.terminate();
            if (loop_thread.joinable()) loop_thread.join();
        }
        llama_backend_free();
    }

    // Full request: parse OpenAI body -> post task -> block on result.
    // GIL is released during the blocking wait so Python/asyncio can run.
    // Note: parse_mu is NOT held here — Tier2 #3 explicitly tests whether
    // oaicompat_chat_params_parse + jinja rendering is safe for concurrent calls.
    std::string generate(const std::string & body_str) {
        json body = json::parse(body_str);

        server_response_reader rd = ctx.get_response_reader();
        server_task task(SERVER_TASK_TYPE_COMPLETION);
        task.id = rd.get_new_id();

        {
            std::vector<raw_buffer> files;
            json parsed = oaicompat_chat_params_parse(body, meta->chat_params, files);
            const llama_vocab * vocab =
                llama_model_get_vocab(llama_get_model(ctx.get_llama_context()));
            task.params = server_schema::eval_llama_cmpl_schema(
                vocab, params, meta->slot_n_ctx, meta->logit_bias_eog, parsed);
            task.params.stream       = false;
            task.params.res_type     = TASK_RESPONSE_TYPE_NONE;
            task.params.cache_prompt = false;  // don't reuse KV across requests
            task.cli              = true;
            task.cli_prompt       = parsed.at("prompt").get<std::string>();
            task.cli_files        = std::move(files);
        }

        rd.post_task(std::move(task));

        std::string content;
        {
            // THE key mechanic: release GIL while blocking on the C++ condvar.
            py::gil_scoped_release release;
            auto r = rd.next([]() { return false; });  // block until result
            if (!r)               throw std::runtime_error("no result (stopped)");
            if (r->is_error())    throw std::runtime_error("task error: " + r->to_json().dump());
            content = r->to_json().value("content", std::string());
        }  // GIL reacquired here; pybind11 builds the Python str with GIL held
        return content;
    }

    // Streaming: yields one token string at a time until the model stops.
    // Each call blocks on rd.next() with GIL released; builds the Python str
    // under GIL. Mirrors what engine.stream()/astream() will need.
    //
    // NOTE: same concurrency caveat as generate() — not thread-safe for
    // concurrent calls on the same Engine (Tier2 #3 finding).
    py::list generate_stream(const std::string & body_str) {
        json body = json::parse(body_str);

        server_response_reader rd = ctx.get_response_reader();
        server_task task(SERVER_TASK_TYPE_COMPLETION);
        task.id = rd.get_new_id();

        {
            std::vector<raw_buffer> files;
            json parsed = oaicompat_chat_params_parse(body, meta->chat_params, files);
            const llama_vocab * vocab =
                llama_model_get_vocab(llama_get_model(ctx.get_llama_context()));
            task.params = server_schema::eval_llama_cmpl_schema(
                vocab, params, meta->slot_n_ctx, meta->logit_bias_eog, parsed);
            task.params.stream       = true;   // <-- streaming on
            task.params.res_type     = TASK_RESPONSE_TYPE_NONE;
            task.params.cache_prompt = false;
            task.cli              = true;
            task.cli_prompt       = parsed.at("prompt").get<std::string>();
            task.cli_files        = std::move(files);
        }

        rd.post_task(std::move(task));

        py::list chunks;
        // states[] is set by post_task; reader's next() will call result->update()
        // to produce incremental diffs. We pull until is_stop() (the final chunk).
        while (true) {
            std::string delta;
            bool is_stop = false;
            {
                py::gil_scoped_release release;
                auto r = rd.next([]() { return false; });
                if (!r)             throw std::runtime_error("stream stopped");
                if (r->is_error())  throw std::runtime_error("stream error: " + r->to_json().dump());
                // partial's first chunk (is_begin) returns null json (HTTP header signal)
                auto j = r->to_json();
                if (!j.is_null()) {
                    delta = j.value("content", std::string());
                }
                is_stop = r->is_stop();
            }
            if (!delta.empty()) chunks.append(delta);
            if (is_stop) break;
        }
        return chunks;
    }
};

PYBIND11_MODULE(mineru_engine_spike, m) {
    m.doc() = "Tier1 #2 spike: async/GIL bridge for MinerU VLM Engine";
    py::class_<Engine>(m, "Engine")
        .def(py::init<const std::string &, const std::string &, int, int, int>(),
             py::arg("model"), py::arg("mmproj"), py::arg("n_ctx") = 8192,
             py::arg("n_gpu_layers") = 99, py::arg("n_parallel") = 1)
        .def("generate", &Engine::generate, py::arg("body"),
             "Run one completion from an OpenAI chat JSON body; blocks until done.")
        .def("generate_stream", &Engine::generate_stream, py::arg("body"),
             "Stream a completion: returns a list of token-string chunks.");
}
