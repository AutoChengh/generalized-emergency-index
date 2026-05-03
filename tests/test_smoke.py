from gei import compute_single_frame


def test_compute_single_frame_smoke():
    result = compute_single_frame(
        504.0451, -271.9787, 22.9184, 2.5530, 17.0237, 2.5907, 0.0,
        501.8724, -278.5692, 24.9702, 2.4877, 16.3289, 2.5973, 0.0,
    )

    assert "GEI" in result
    assert result["GEI"] >= 0
