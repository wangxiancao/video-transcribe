from pipeline.asr.glm import _chunks


def test_short_tail_is_merged_into_previous_chunk():
    samples = [0.0] * int(30.016 * 16000)

    chunks = _chunks(samples, 16000, 30)

    assert len(chunks) == 1
    assert len(chunks[0]) == int(30.016 * 16000)


def test_short_tail_after_full_chunks_is_merged():
    samples = [0.0] * int(60.016 * 16000)

    chunks = _chunks(samples, 16000, 30)

    assert [len(chunk) for chunk in chunks] == [480000, int(30.016 * 16000)]


def test_normal_full_chunks_are_unchanged():
    samples = [0.0] * (16000 * 60)

    chunks = _chunks(samples, 16000, 30)

    assert [len(chunk) for chunk in chunks] == [480000, 480000]

def test_engine_windows_split_inside_envelope():
    from pipeline.asr.native import MAX_ENGINE_CHUNKS, _engine_windows

    # single window below the envelope
    assert _engine_windows(100) == [(0, 100)]
    assert _engine_windows(MAX_ENGINE_CHUNKS) == [(0, MAX_ENGINE_CHUNKS)]
    # the 117-min real case: 235 chunks -> 120 + 115
    assert _engine_windows(235) == [(0, 120), (120, 235)]
    # exact multiples tile cleanly
    assert _engine_windows(240) == [(0, 120), (120, 240)]
    # every window respects the envelope and windows tile without overlap
    for total in (1, 119, 121, 235, 240, 361):
        windows = _engine_windows(total)
        assert windows[0][0] == 0 and windows[-1][1] == total
        assert all(end - start <= MAX_ENGINE_CHUNKS for start, end in windows)
        assert all(b == a for (_, b), (a, _) in zip(windows, windows[1:]))
