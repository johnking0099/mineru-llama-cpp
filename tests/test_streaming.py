from mineru_llama_cpp import GenerateChunk, SamplingParams


def test_stream_yields_multiple_chunks_with_final_metadata(engine):
    chunks = list(engine.stream([{"role": "user", "content": "List three fruits, one per line."}]))
    assert len(chunks) > 1
    assert all(isinstance(c, GenerateChunk) for c in chunks)
    assert all(c.finish_reason is None for c in chunks[:-1])
    assert chunks[-1].finish_reason is not None
    assert chunks[-1].timings is not None


def test_stream_concatenation_matches_generate(engine):
    messages = [{"role": "user", "content": "List three fruits, one per line."}]
    streamed = "".join(c.delta for c in engine.stream(messages))
    non_streamed = engine.generate(messages).content
    assert streamed == non_streamed


def test_stream_chunk_count_grows_with_n_predict(engine):
    messages = [{"role": "user", "content": "List three fruits, one per line."}]
    short = list(engine.stream(messages, SamplingParams(temperature=0.0, top_k=1, n_predict=32)))
    long_ = list(engine.stream(messages, SamplingParams(temperature=0.0, top_k=1, n_predict=96)))
    assert len(long_) >= len(short)


async def test_astream_yields_same_content_as_stream(engine):
    messages = [{"role": "user", "content": "List three fruits, one per line."}]
    sp = SamplingParams(temperature=0.0, top_k=1, n_predict=32)

    async_chunks = [c async for c in engine.astream(messages, sp)]
    sync_chunks = list(engine.stream(messages, sp))

    assert "".join(c.delta for c in async_chunks) == "".join(c.delta for c in sync_chunks)
    assert async_chunks[-1].finish_reason is not None
