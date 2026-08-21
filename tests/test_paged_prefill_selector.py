from nanovllm.layers.attention import Attention


def _attention(**config):
    return Attention(
        num_heads=4,
        head_dim=64,
        scale=64**-0.5,
        num_kv_heads=2,
        split_kv_config=config,
    )


def test_selector_explicit_modes_override_auto():
    assert _attention(paged_prefill_q_tile_mode="serial")._select_paged_prefill_path(64, 16384, 32) is False
    assert _attention(paged_prefill_q_tile_mode="parallel")._select_paged_prefill_path(1, 32, 32) is True


def test_selector_auto_short_long_batch_and_tile_boundary():
    selector = _attention(
        paged_prefill_q_tile_mode="auto",
        paged_prefill_auto_min_batch_size=8,
        paged_prefill_auto_min_q_tiles=128,
        paged_prefill_auto_parallel_enabled=False,
    )
    assert selector._select_paged_prefill_path(1, 4096, 32) is False
    assert selector._select_paged_prefill_path(7, 4096, 32) is False
    assert selector._select_paged_prefill_path(8, 4064, 32) is False
    assert selector._select_paged_prefill_path(8, 4096, 32) is False
    assert selector._select_paged_prefill_path(8, 4096, 64) is False
    revalidated = _attention(
        paged_prefill_q_tile_mode="auto",
        paged_prefill_auto_min_batch_size=8,
        paged_prefill_auto_min_q_tiles=128,
        paged_prefill_auto_parallel_enabled=True,
    )
    assert revalidated._select_paged_prefill_path(8, 4096, 32) is True


def test_selector_legacy_bool_forces_path():
    assert _attention(paged_prefill_q_tile_parallel=False)._select_paged_prefill_path(8, 16384, 32) is False
    assert _attention(paged_prefill_q_tile_parallel=True)._select_paged_prefill_path(1, 32, 32) is True
