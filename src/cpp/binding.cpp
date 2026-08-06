// binding.cpp — thin pybind11 layer. GIL release/acquire, exception-type
// mapping, and dict marshalling only. No business logic (design spec §4.1).
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <nlohmann/json.hpp>

#include "engine_core.h"

#include <string>

namespace py = pybind11;
using json = nlohmann::ordered_json;

namespace {

// Raises the appropriate Python exception subclass (imported from
// mineru_llama_cpp.exceptions) and never returns. `type` is one of
// llama-server's error_type strings (see server-common.cpp
// format_error_response()), or "invalid_request_error" for pre-post C++
// exceptions caught in the binding layer.
[[noreturn]] void raise_mapped_error(const std::string & type, const std::string & message) {
    py::object exceptions_mod = py::module_::import("mineru_llama_cpp.exceptions");
    py::object exc_class;
    if (type == "exceed_context_size_error") {
        exc_class = exceptions_mod.attr("ContextExceededError");
    } else if (type == "invalid_request_error") {
        exc_class = exceptions_mod.attr("InvalidRequestError");
    } else {
        exc_class = exceptions_mod.attr("EngineError");
    }
    PyErr_SetString(exc_class.ptr(), message.c_str());
    throw py::error_already_set();
}

// Parses an EngineCore error_json string (see server-common.cpp
// format_error_response(): {"code":int,"message":str,"type":str}) and
// raises the mapped exception. Falls back to EngineError with the raw
// string as the message if error_json isn't valid JSON (defensive; should
// not happen in practice since it always originates from
// format_error_response()).
[[noreturn]] void raise_from_error_json(const std::string & error_json_str) {
    std::string type    = "server_error";
    std::string message = error_json_str;
    try {
        json ej = json::parse(error_json_str);
        type    = ej.value("type", type);
        message = ej.value("message", error_json_str);
    } catch (...) {
        // leave type/message at their fallback values
    }
    raise_mapped_error(type, message);
}

py::dict timings_to_dict(const EngineCore::Timings & t) {
    py::dict d;
    d["prompt_n"]             = t.prompt_n;
    d["prompt_ms"]            = t.prompt_ms;
    d["prompt_per_second"]    = t.prompt_per_second;
    d["predicted_n"]          = t.predicted_n;
    d["predicted_ms"]         = t.predicted_ms;
    d["predicted_per_second"] = t.predicted_per_second;
    return d;
}

py::dict generate_impl(EngineCore & self, const std::string & body) {
    EngineCore::GenerateResult r;
    try {
        py::gil_scoped_release release;
        r = self.generate(body);
    } catch (const std::exception & e) {
        raise_mapped_error("invalid_request_error", e.what());
    }
    if (r.is_error) {
        raise_from_error_json(r.error_json);
    }
    py::dict out;
    out["content"]          = r.content;
    out["finish_reason"]    = r.finish_reason;
    out["tokens_evaluated"] = r.tokens_evaluated;
    out["tokens_predicted"] = r.tokens_predicted;
    out["timings"]          = timings_to_dict(r.timings);
    return out;
}

// Wraps EngineCore::StreamHandle (a move-only C++ type) so pybind11 can
// hold it inside a Python-iterable object. Implements the Python iterator
// protocol: __next__ returns a dict per chunk (including the final one,
// which carries finish_reason/tokens_*/timings) and raises StopIteration on
// the call *after* the final chunk was returned.
class PyStreamIterator {
public:
    explicit PyStreamIterator(EngineCore::StreamHandle handle) : handle_(std::move(handle)) {}

    py::dict next() {
        if (finished_) {
            throw py::stop_iteration();
        }
        EngineCore::Chunk c;
        {
            py::gil_scoped_release release;
            c = handle_.next_chunk();
        }
        if (c.is_error) {
            raise_from_error_json(c.error_json);
        }
        py::dict out;
        out["delta"] = c.delta;
        if (c.is_final) {
            finished_ = true;
            out["finish_reason"]    = c.finish_reason;
            out["tokens_evaluated"] = c.tokens_evaluated;
            out["tokens_predicted"] = c.tokens_predicted;
            out["timings"]          = timings_to_dict(c.timings);
        } else {
            out["finish_reason"]    = py::none();
            out["tokens_evaluated"] = py::none();
            out["tokens_predicted"] = py::none();
            out["timings"]          = py::none();
        }
        return out;
    }

private:
    EngineCore::StreamHandle handle_;
    bool finished_ = false;
};

PyStreamIterator generate_stream_impl(EngineCore & self, const std::string & body) {
    // No GIL release here: this only does the fast parse+post phase (a few
    // ms at most for jinja rendering), not the blocking wait — matches the
    // parse_mu_-guarded critical section in EngineCore. next_chunk() (above)
    // is what releases the GIL for the actual blocking wait.
    try {
        return PyStreamIterator(self.generate_stream(body));
    } catch (const std::exception & e) {
        raise_mapped_error("invalid_request_error", e.what());
    }
}

} // namespace

PYBIND11_MODULE(_mineru_llama_cpp, m) {
    py::class_<PyStreamIterator>(m, "_StreamIterator")
        .def("__iter__", [](PyStreamIterator & self) -> PyStreamIterator & { return self; })
        .def("__next__", &PyStreamIterator::next);

    py::class_<EngineCore>(m, "_EngineCore")
        .def(py::init<const std::string &, const std::string &, int, int, int, int32_t, int32_t>(),
             py::arg("model_path"), py::arg("mmproj_path"), py::arg("n_ctx_seq"),
             py::arg("n_gpu_layers"), py::arg("n_parallel"), py::arg("verbosity"), py::arg("n_threads"))
        .def("generate", &generate_impl, py::arg("body"))
        .def("generate_stream", &generate_stream_impl, py::arg("body"))
        .def_property_readonly("eos_token_str", &EngineCore::eos_token_str);
}
