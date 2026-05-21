from app.services.postprocess import PostprocessState, smooth_prediction


def test_low_confidence_does_not_emit() -> None:
    st = PostprocessState()
    text, updated = smooth_prediction("hello", 0.1, st, min_confidence=0.2, cooldown_seconds=1.0)
    assert text == ""
    assert not updated


def test_valid_prediction_emits() -> None:
    st = PostprocessState()
    text, updated = smooth_prediction("Hello", 0.8, st, min_confidence=0.2, cooldown_seconds=1.0)
    assert text == "hello"
    assert updated
