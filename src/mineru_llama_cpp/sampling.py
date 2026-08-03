"""Sampling parameters for mineru_llama_cpp.Engine. See design spec §5.2.

Field names are copied verbatim from the JSON keys registered in
llama.cpp's tools/server/server-schema.cpp (field_num/field_json calls) —
NOT from vLLM/HuggingFace naming conventions. Notably: `repeat_penalty`
(not `repetition_penalty`), `n_predict` (not `max_tokens`, though
`max_tokens`/`max_completion_tokens` are llama.cpp's own OpenAI-compatible
aliases for n_predict and would also work if sent directly as JSON, but the
canonical field here is the native name).

Deliberately excluded (not oversights — see design spec §5.2): grammar,
json_schema, logit_bias, samplers, adaptive_target, adaptive_decay,
backend_sampling, post_sampling_probs. These are advanced/structured-output
features orthogonal to basic sampling.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class SamplingParams:
    # Generation length
    n_predict: int | None = None  # alias: max_tokens, max_completion_tokens

    # Core sampling
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

    # Repetition penalties
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

    # Other
    seed: int | None = None
    stop: list[str] | None = None
    n_probs: int | None = None
    min_keep: int | None = None
    ignore_eos: bool | None = None

    def to_json_fields(self) -> dict:
        """Returns a dict of only the explicitly-set (non-None) fields,
        suitable for merging into an OpenAI-style chat completion body."""
        return {k: v for k, v in asdict(self).items() if v is not None}
