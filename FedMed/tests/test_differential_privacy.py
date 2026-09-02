import numpy as np

from security.differential_privacy import (
    add_gaussian_noise,
    clip_update,
)


def test_clip_update_limits_norm():
    update = {
        "layer": np.array([3.0, 4.0]),
    }

    clipped = clip_update(update, max_norm=1.0)

    norm = np.linalg.norm(clipped["layer"])

    assert norm <= 1.0 + 1e-6


def test_small_update_is_not_changed_by_clipping():
    update = {
        "layer": np.array([0.3, 0.4]),
    }

    clipped = clip_update(update, max_norm=1.0)

    np.testing.assert_allclose(
        clipped["layer"],
        update["layer"],
    )


def test_gaussian_noise_changes_update():
    update = {
        "layer": np.zeros(100),
    }

    protected = add_gaussian_noise(
        update,
        noise_multiplier=0.1,
        max_norm=1.0,
        seed=42,
    )

    assert not np.array_equal(
        protected["layer"],
        update["layer"],
    )


def test_zero_noise_preserves_clipped_update():
    update = {
        "layer": np.array([3.0, 4.0]),
    }

    protected = add_gaussian_noise(
        update,
        noise_multiplier=0.0,
        max_norm=1.0,
        seed=42,
    )

    expected = clip_update(update, max_norm=1.0)

    np.testing.assert_allclose(
        protected["layer"],
        expected["layer"],
    )