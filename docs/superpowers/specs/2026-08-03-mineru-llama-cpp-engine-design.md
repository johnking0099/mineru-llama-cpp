# mineru-llama-cpp Engine 设计

> 状态：设计定案，待用户审阅
> 前置：`docs/stage0-lib-feasibility.md`（lib 化可行性）、`docs/stage1a-path-a-verification.md`（路 A / async-GIL / 多 slot / 流式 / 错误隔离 / 长驻 / 可复现性，全部技术验证通过）

## 1. 背景与目标

技术验证阶段（阶段 0 + 阶段 1a，Tier 1/2/3-部分）已证明：llama-server 核心可 lib 化、JSON/oaicompat 路径可脱离 server-http 复用、async/GIL 桥接模型成立、多 slot 并发可用、流式可用、错误隔离可用、长驻服务可用、temp=0 可复现。架构级风险（Tier 1+2）已全部清零。

本轮目标：把验证过的能力实现成一个正式的、可发布的 Python 库 **`mineru-llama-cpp`**（原代号 `mineru-vl-engine`），供 `mineru-vl-utils` 未来集成为一个新的 `VlmClient` backend（类似现有的 `vllm-engine`/`vllm-async-engine`）。

### 非目标（本轮不做）

- 跨平台 wheel 打包 / CI（`cibuildwheel`、多平台 GitHub Actions）——留给后续阶段。
- CUDA/Vulkan 后端——本机只有 Metal，其余后端未验证。
- HuggingFace repo id 自动下载模型——只接受本地文件路径。
- `batch_generate()` 专用接口——批量由调用方并发 `agenerate()` 实现。
- `priority` 请求优先级参数——未验证过 llama-server 的 `post_task(front=true)` 机制。
- BF16+Metal 崩溃（上游 bug #21381）的自动检测/拦截——只在文档提醒，不写防护代码。
- mineru-vl-utils 侧的 `VlmClient` backend 实现——这是下一个独立项目，本轮只产出被集成的库。

## 2. 提供侧接口调研结论

对比 vLLM（`LLM`/`AsyncLLM`）、lmdeploy（`AsyncEngine`/`Pipeline`）、mlx-vlm（模块级函数）、transformers（`model.generate`）四个参照库的**原生**接口（非 mineru-vl-utils 包装层），关键结论：

| 维度 | 参照库共识 | 本设计选择 |
|---|---|---|
| 引擎对象 | vLLM/lmdeploy 有独立引擎类；mlx-vlm 无状态函数；transformers 引擎即模型对象 | 有独立 `Engine` 类（自己 `load_model`，是唯一"引擎在库内创建"的情况） |
| 流式 vs 非流式 | vLLM AsyncLLM 本身即流式（无独立非流式方法）；lmdeploy 一个方法 + bool 参数切换；mlx-vlm 两个独立函数 | 两个独立方法（`generate`/`stream`），与 mlx-vlm 同构 |
| 采样参数 | vLLM/lmdeploy/transformers 用独立配置对象；mlx-vlm 用裸 kwargs | 独立对象 `SamplingParams`，与主流一致 |
| 批处理原语 | vLLM `LLM.generate`/transformers 接口层原生支持批；vLLM `AsyncLLM`/lmdeploy `AsyncEngine.generate` 无批，靠并发调度 | 无批处理原语，与 vLLM `AsyncLLM`/lmdeploy `AsyncEngine` 同构（对应 server_context 多 slot 在 C++ 内部拼批） |
| 返回类型 | 四个库全部返回带 metadata 的结构体（`RequestOutput`/`GenOut`/`GenerationResult`），仅 transformers 返回裸 tensor | 返回带 metadata 的 `GenerateResult`/`GenerateChunk`，与主流一致 |
| 图片输入 | vLLM/lmdeploy 接受 `PIL.Image`；mlx-vlm 接受路径字符串（内部自解码） | 只接受 base64 data URI 字符串，比四个参照库都更严格（刻意收窄，见 §6） |

## 3. 仓库与目录结构

新建独立 git 仓库 `mineru-llama-cpp`（当前 `mineru-vl-engine` 的技术验证代码将迁移进来）。

```
mineru-llama-cpp/                      # 新 git 仓库根目录
├── .gitmodules                        # third_party/llama.cpp submodule
├── pyproject.toml                     # scikit-build-core 构建后端
├── CMakeLists.txt                     # 顶层 CMake
├── third_party/
│   └── llama.cpp/                     # git submodule，固定 commit 9a3bf2b84
├── src/
│   ├── mineru_llama_cpp/              # Python 包（下划线，import 名）
│   │   ├── __init__.py                # 导出 Engine, SamplingParams, 异常类, 类型
│   │   ├── engine.py                  # Engine（同步+异步方法并存）
│   │   ├── sampling.py                # SamplingParams dataclass
│   │   ├── types.py                   # Message/ContentPart/ImageURL 等 TypedDict + GenerateResult/GenerateChunk
│   │   └── exceptions.py              # 异常层级
│   └── cpp/
│       ├── binding.cpp                # pybind11 绑定入口：GIL release/acquire、异常映射、类型转换
│       ├── engine_core.h/.cpp         # C++ Engine 核心：server_context + oaicompat 解析 + 锁
│       └── CMakeLists.txt
├── tests/
│   ├── legacy_spike/                  # 归档：mineru_engine_spike.cpp, metal_probe.mm, test_*.py（技术验证阶段产物，只读参考）
│   ├── conftest.py                    # session 级 engine fixture（避免重复加载模型）
│   ├── test_generate.py               # generate/agenerate（文本+图片）
│   ├── test_streaming.py              # stream/astream 逐 chunk
│   ├── test_concurrency.py            # 多 slot 并发 + 路由正确性
│   ├── test_error_handling.py         # 异常层级 + 错误隔离
│   ├── test_lifecycle.py              # 长驻 + 优雅关闭
│   └── test_determinism.py            # temp=0 可复现性
└── docs/
    ├── api.md
    └── known-issues.md                # BF16+Metal 崩溃提醒（llama.cpp#21381）
```

包名细节：pip/仓库名 `mineru-llama-cpp`（带短杠），Python `import mineru_llama_cpp`（下划线）。

## 4. C++ 层设计

### 4.1 分层：绑定层 vs 核心层

- **`binding.cpp`**（薄）：只做 pybind11 类型转换、GIL release/acquire、把核心层返回的 `is_error`/`error_json` 映射成对应的 Python 异常子类。不含业务逻辑，未来加 CUDA/Vulkan 变体时不用改。
- **`engine_core.h/.cpp`**（核心）：装 `server_context`、oaicompat 解析（路 A）、并发锁、流式迭代器。复用技术验证阶段验证过的全部逻辑。

### 4.2 `EngineCore` 接口

```cpp
class EngineCore {
public:
    EngineCore(const std::string& model_path, const std::string& mmproj_path,
               int n_ctx, int n_gpu_layers, int n_parallel);
    ~EngineCore();  // terminate() + join() + backend_free()

    struct GenerateResult {
        std::string content;
        std::string finish_reason;   // "stop" | "length"，由 stop_type 映射而来
        int32_t tokens_evaluated;
        int32_t tokens_predicted;
        // timings: prompt_n/prompt_ms/prompt_per_second/predicted_n/predicted_ms/predicted_per_second
        double prompt_ms, prompt_per_second, predicted_ms, predicted_per_second;
        int32_t prompt_n, predicted_n;
        bool is_error;
        std::string error_json;      // send_error 产生的 JSON，供 binding 层分类映射异常
    };
    GenerateResult generate(const std::string& body_json);

    class StreamHandle {
    public:
        struct Chunk {
            std::string delta;
            bool is_final;           // true 时下面字段才有效
            std::string finish_reason;
            int32_t tokens_evaluated, tokens_predicted;
            double prompt_ms, prompt_per_second, predicted_ms, predicted_per_second;
            int32_t prompt_n, predicted_n;
            bool is_error;
            std::string error_json;
        };
        Chunk next_chunk();  // 阻塞到下一个 token 或结束；不持有 EngineCore 的锁
    private:
        server_response_reader rd_;
    };
    StreamHandle generate_stream(const std::string& body_json);

private:
    server_context ctx_;
    common_params params_;
    std::unique_ptr<server_context_meta> meta_;
    std::thread loop_thread_;
    std::mutex parse_mu_;  // 只包解析+post阶段；不包阻塞等待，多 slot 并行 decode 不受影响
};
```

**关键设计点**（对应技术验证阶段发现的 bug 修复）：

1. **锁粒度**：`parse_mu_` 只包 `oaicompat_chat_params_parse` + `eval_llama_cmpl_schema` + `post_task`。修复验证阶段发现的"并发调用竞态导致 empty prompt"问题（Tier2 #3），同时不牺牲多 slot 并行 decode 的性能。
2. **`cache_prompt = false`** 硬编码在 `generate`/`generate_stream` 内部，不作为可调参数暴露。修复验证阶段发现的"cache_prompt 默认 true 导致连续请求 empty prompt"问题。
3. **错误跨边界传递用 `is_error`/`error_json` 字段，不是裸 C++ 异常**。这样 binding 层能精确区分 server 层错误（`send_error` 产生，可能是"图片解码失败"或"超 ctx"）并映射到对应的 Python 异常子类；C++ 侧真正的 `throw`（如 `json::parse` 失败）仍直接抛，pybind11 转译成 `InvalidRequestError`。
4. **`finish_reason` 映射**：`stop_type_to_str()` 返回 "eos"/"word"/"limit"/"none"；binding 层映射 "eos"/"word" → `"stop"`，"limit" → `"length"`（复刻 llama.cpp 自己在 oaicompat 路径里已有的同一段映射逻辑，非发明）。
5. **流式用 `StreamHandle`/迭代器而非 callback**：binding 层包装成 Python 生成器，每次 `__next__` 调 `next_chunk()`，内部 `gil_scoped_release`。

## 5. Python 层设计

### 5.1 `types.py` — 输入/输出类型

```python
from typing import Literal, TypedDict
from dataclasses import dataclass


# --- 输入：messages ---

class ImageURL(TypedDict):
    url: str
    """base64 data URI 字符串，如 "data:image/png;base64,...."。
    不接受本地路径/HTTP URL/裸 base64（无 data: 前缀）。"""


class TextPart(TypedDict):
    type: Literal["text"]
    text: str


class ImagePart(TypedDict):
    type: Literal["image_url"]
    image_url: ImageURL


ContentPart = TextPart | ImagePart


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str | list[ContentPart]


Messages = list[Message]


# --- 输出：generate() / stream() ---

@dataclass(frozen=True)
class GenerationTimings:
    prompt_n: int
    prompt_ms: float
    prompt_per_second: float
    predicted_n: int
    predicted_ms: float
    predicted_per_second: float


@dataclass(frozen=True)
class GenerateResult:
    content: str
    finish_reason: Literal["stop", "length"]
    tokens_evaluated: int
    tokens_predicted: int
    timings: GenerationTimings


@dataclass(frozen=True)
class GenerateChunk:
    delta: str
    finish_reason: Literal["stop", "length"] | None = None
    tokens_evaluated: int | None = None
    tokens_predicted: int | None = None
    timings: GenerationTimings | None = None
```

- `role` 只含 `"system"/"user"/"assistant"`——不做 function calling，没有 `"tool"` 角色。
- `content` 允许 `str` 直传（纯文本请求不强制包 list）或 `list[ContentPart]`。
- `ContentPart` 只有 `text`/`image_url` 两种——mtmd 只验证过图片。
- `GenerateChunk` 中间态只有 `delta` 非空，其余 `None`；终态 chunk 通过 `finish_reason is not None` 判断，同时携带完整 metadata（对应 llama.cpp 流式模式下最终 chunk 的 `content` 字段本身也是增量而非全文的实测行为）。
- `content`（全量）vs `delta`（增量）故意用不同字段名区分语义，参照 OpenAI SDK 的 `message.content` vs `delta.content` 命名惯例。

### 5.2 `sampling.py` — `SamplingParams`

暴露 llama.cpp 支持的采样参数，**字段名逐一核对自 `server-schema.cpp` 里 `field_num`/`field_json` 注册的 JSON key**（这些 key 就是 oaicompat body 实际接受的名字），不是凭 vLLM/HF 命名习惯猜的——核对时纠正了两处：`repeat_penalty`（不是 `repetition_penalty`）、生成长度用原生名 `n_predict`（`max_tokens`/`max_completion_tokens` 是 llama.cpp 自己注册的 OpenAI 兼容 alias，序列化时两者等价，字段名选原生的）：

```python
@dataclass
class SamplingParams:
    # 生成长度
    n_predict: int | None = None          # alias: max_tokens, max_completion_tokens

    # 核心采样
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    typical_p: float | None = None
    top_n_sigma: float | None = None
    xtc_probability: float | None = None
    xtc_threshold: float | None = None
    dynatemp_range: float | None = None
    dynatemp_exponent: float | None = None

    # 重复惩罚
    repeat_last_n: int | None = None
    repeat_penalty: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None

    # DRY
    dry_multiplier: float | None = None
    dry_base: float | None = None
    dry_allowed_length: int | None = None
    dry_penalty_last_n: int | None = None
    dry_sequence_breakers: list[str] | None = None

    # Mirostat
    mirostat: int | None = None
    mirostat_tau: float | None = None
    mirostat_eta: float | None = None

    # 其他
    seed: int | None = None
    stop: list[str] | None = None
    n_probs: int | None = None
    min_keep: int | None = None
    ignore_eos: bool | None = None
```

**明确排除**（不是遗漏，是刻意不做）：`grammar`/`json_schema`（约束生成/结构化输出）、`logit_bias`（逐 token 偏置）、`samplers`（采样器顺序自定义）、`adaptive_target`/`adaptive_decay`（自适应采样，llama.cpp 较新且冷门的实验特性）、`backend_sampling`/`post_sampling_probs`（后端采样开关/概率回传，非采样行为本身）。这些是独立的高级特性而非基础采样参数，v1 不暴露；需要时可作为 `SamplingParams` 之外的独立参数或后续版本再加，不影响本设计的其余部分。

### 5.3 `exceptions.py` — 分层异常

```python
class MineruLlamaCppError(Exception):
    """所有本库异常的基类。"""

class InvalidRequestError(MineruLlamaCppError):
    """请求本身有问题：malformed JSON、图片解码失败等 400 类错误。"""

class ContextExceededError(InvalidRequestError):
    """prompt 超过 n_ctx。InvalidRequestError 的子类。"""

class EngineError(MineruLlamaCppError):
    """引擎内部错误：load_model 失败、C++ 侧未分类的错误。"""
```

binding 层负责把 `EngineCore::GenerateResult.error_json` 解析并映射成 Python 异常子类。`error_json` 的形状由 llama-server 自己的 `format_error_response()` 决定（`server-common.cpp`），是固定结构，不是要去匹配 message 文本猜出来的：

```json
{"code": 400, "message": "...", "type": "exceed_context_size_error"}
```

`type` 字段是一个封闭枚举（对应 C++ `enum error_type`），映射表：

| `type` 字符串 | Python 异常 |
|---|---|
| `exceed_context_size_error` | `ContextExceededError` |
| `invalid_request_error` | `InvalidRequestError` |
| 其余（`server_error`/`not_found_error`/`permission_error`/`authentication_error`/`not_supported_error`/`unavailable_error`，或未识别值） | `EngineError` |

binding 层直接 switch 这个 `type` 字符串做映射，不做 message 子串匹配（更精确、不随 llama.cpp 措辞变化而失效）。C++ `throw`（body JSON 本身解析失败，发生在 post task 之前，不经过 `format_error_response`）→ pybind11 转译，binding 层统一包成 `InvalidRequestError`。

### 5.4 `engine.py` — `Engine` 类（唯一的类，同步+异步方法并存）

```python
from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import AsyncIterator, Iterator

from .sampling import SamplingParams
from .types import GenerateChunk, GenerateResult, Messages


class Engine:
    def __init__(
        self,
        model: str | Path,
        mmproj: str | Path,
        *,
        n_ctx: int = 8192,
        n_gpu_layers: int = 99,
        n_parallel: int = 1,
    ) -> None:
        """加载模型，启动后台 decode 循环线程。model/mmproj 只接受本地文件路径。"""
        ...

    # --- 非流式 ---

    def generate(
        self,
        messages: Messages,
        sampling_params: SamplingParams | None = None,
    ) -> GenerateResult:
        """同步阻塞调用。

        Raises:
            InvalidRequestError: 请求本身有问题（图片解码失败等）。
            ContextExceededError: prompt 超过 n_ctx（InvalidRequestError 子类）。
            EngineError: 引擎内部错误。
        """
        ...

    async def agenerate(
        self,
        messages: Messages,
        sampling_params: SamplingParams | None = None,
    ) -> GenerateResult:
        """generate() 的异步版本，经 run_in_executor 桥接，不阻塞事件循环。"""
        ...

    # --- 流式 ---

    def stream(
        self,
        messages: Messages,
        sampling_params: SamplingParams | None = None,
    ) -> Iterator[GenerateChunk]:
        """同步生成器，逐 chunk yield。终态 chunk 的 finish_reason 非 None。"""
        ...

    def astream(
        self,
        messages: Messages,
        sampling_params: SamplingParams | None = None,
    ) -> AsyncIterator[GenerateChunk]:
        """stream() 的异步版本，真流式（后台线程 + asyncio.Queue 桥接 call_soon_threadsafe）。"""
        ...

    # --- 生命周期 ---

    def close(self) -> None:
        """显式关闭：terminate + join 后台线程 + 释放 llama backend。幂等。"""
        ...

    async def aclose(self) -> None:
        """close() 的异步版本（offload 到 executor，避免阻塞事件循环）。"""
        ...

    def __enter__(self) -> Engine: ...
    def __exit__(
        self, exc_type: type[BaseException] | None,
        exc_value: BaseException | None, tb: TracebackType | None,
    ) -> None:
        self.close()

    async def __aenter__(self) -> Engine: ...
    async def __aexit__(
        self, exc_type: type[BaseException] | None,
        exc_value: BaseException | None, tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    def __del__(self) -> None:
        """兜底安全网，非主要关闭路径——正常应用应显式调用 close()/aclose()。"""
        ...
```

不使用 `aio_` 前缀（mineru-vl-utils 内部惯例），本库自洽即可，异步方法统一用 `a` 前缀（`agenerate`/`astream`/`aclose`），与 `__aenter__`/`__aexit__` 的 Python 惯例一致。

`astream()` 桥接细节：`run_in_executor` 适合"调一次拿一个结果"，不能直接用在生成器上。实现方式：起一个后台线程跑同步 `for chunk in self.stream(...)`，每收到一个 chunk 用 `asyncio.Queue` + `loop.call_soon_threadsafe` 传回事件循环，`astream()` 从 queue 里 `await get()`。这是同步迭代器转异步迭代器的标准桥接模式。

**`close()`/`aclose()` 与在途请求的关系**：v1 约定调用方在没有在途请求时才调用 `close()`（技术验证阶段 Tier2 #6 验证的是"跑完 100 个请求后再关"这一种场景，没有验证过"关闭时有请求正在 decode"的行为）。`close()` 内部 `ctx.terminate()` 会让 loop 线程退出，此时若有在途 `generate()`/`stream()` 正阻塞等待结果，其 `rd.next()` 会因为 loop 停止而永久不返回或返回错误——v1 不专门处理这个竞争场景（不属于本轮已验证范围），留给后续版本按需加"等待在途请求完成再关闭"的排水逻辑。

## 6. 图片输入格式（刻意收窄，非遗漏）

`ImagePart.image_url.url` **只接受 base64 data URI 字符串**（如 `"data:image/png;base64,...."`），不接受 `PIL.Image` 对象、本地路径、HTTP URL、裸 base64（无 `data:` 前缀）。

三层理由：
1. mineru-vl-utils 的 `http_client.py` 已经在做同样的 base64 编码（`get_image_data_url`）——它未来调用本库时直接复用现成代码，零新增负担。
2. 接受 `PIL.Image` 需要引入 `pillow` 依赖并处理格式探测/RGBA 转换，这些逻辑在 mineru-vl-utils 里已有正确实现，重复维护是负资产，违反"薄+少+抗升级"原则。
3. 本库的依赖底线是 `dependencies = []`（纯 pybind11 + 标准库），维持这条底线本身就是设计目标之一。

编码这一层职责完全留给上层，本层保持绝对薄、绝对单一。

## 7. 构建系统

- `pyproject.toml`：`build-backend = "scikit_build_core.build"`，`requires = ["scikit-build-core>=0.10", "pybind11"]`。选 scikit-build-core 是为将来 wheel 打包铺路（虽然本轮不做跨平台 CI）。
- 顶层 `CMakeLists.txt`：`add_subdirectory(third_party/llama.cpp)` + `add_subdirectory(src/cpp)`。与技术验证阶段的 spike 不同——spike 链接的是预编译好的 `llama.cpp-build/llama.cpp/build` 产物，正式库改成从 submodule 源码完整编译（包括 Metal shader），首次构建会明显变慢（几分钟级），这是 submodule 化的预期代价，非异常。
- `GGML_METAL` 编译选项依赖 llama.cpp 自身 CMake 的平台自动检测，顶层不显式指定，为将来切 Linux/CUDA 时减少顶层配置改动。

## 8. 测试策略

技术验证阶段的 spike 代码（`mineru_engine_spike.cpp`、`metal_probe.mm`、全部 `test_*.py`）整体归档到 `tests/legacy_spike/`，作为参考/回归基准保留，不直接复用为正式测试（正式 `EngineCore` 会有新的类型/接口）。

正式测试改写为 pytest，覆盖技术验证阶段已验证过的全部场景：

| 测试文件 | 对应验证阶段发现 |
|---|---|
| `test_generate.py` | 路 A（JSON/oaicompat）、BF16+Metal 崩溃回避（用 Q8_0 模型） |
| `test_streaming.py` | 流式 partial result，chunk 数随 max_tokens 增长，拼接一致性 |
| `test_concurrency.py` | 多 slot 并发 speedup + 路由正确性（`conc[i]==serial[i]`） |
| `test_error_handling.py` | 错误隔离 + 新增：`pytest.raises(InvalidRequestError)`/`pytest.raises(ContextExceededError)` 分层异常断言 |
| `test_lifecycle.py` | 长驻 100 请求 RSS 稳定、`close()` 干净释放、可重建 |
| `test_determinism.py` | temp=0 + seed 跨次/跨 slot 逐字节一致 |

`conftest.py` 用 `scope="session"` fixture 复用 engine 实例，避免每个测试重新加载 1.4GB 模型。模型路径写死在测试里（本轮不处理模型分发，同技术验证阶段）。

## 9. 与 mineru-vl-utils 集成的前瞻说明（非本轮范围，仅记录）

- 未来 mineru-vl-utils 需要给 `VlmClient` 抽象类补上 `stream`/`astream` 抽象方法——当前该抽象类的五个现有 backend（http-client/vllm-engine/vllm-async-engine/transformers/mlx-engine）均无流式对外接口，本库是第一个提供真流式的，需要上层抽象跟进。
- `ImageType`（mineru-vl-utils 的 `PIL.Image|bytes|str|Sequence`）到本库 `str`（data URI）的转换，直接复用 `http_client.py` 的 `get_image_data_url`。
- `backend` 枚举字符串（如 `"llama-cpp-engine"`）、`batching_mode` 归类（"concurrent" 还是 "stepping"——本库因为有真多 slot 并发，应归 "concurrent"，与 vllm-async-engine 同类）等，留给该集成项目自己的设计阶段决定。
