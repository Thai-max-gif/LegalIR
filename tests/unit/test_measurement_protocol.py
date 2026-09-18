"""F5: local benchmark/measurement protocol primitives."""
from src.pipeline.kaggle_train import (
    StageTimingTelemetry,
    resource_inventory,
)


def test_stage_telemetry_records_exclusive_seconds_and_cache_flag():
    tel = StageTimingTelemetry()
    tel.record("a", elapsed_seconds=1.5, cache_hit=False, workload_count=10)
    tel.record("b", elapsed_seconds=2.5, cache_hit=True, workload_count=20)
    d = tel.to_dict()
    assert d["a"]["seconds"] == 1.5 and d["a"]["cache_hit"] is False
    assert d["b"]["seconds"] == 2.5 and d["b"]["cache_hit"] is True
    assert d["a"]["workload_count"] == 10
    # Exclusive sequential stages must not double-count: sum equals total.
    assert d["a"]["seconds"] + d["b"]["seconds"] == 4.0


def test_stage_telemetry_cache_hit_distinguishes_warm_from_cold():
    tel = StageTimingTelemetry()
    tel.record("dense_index", elapsed_seconds=0.01, cache_hit=True)
    assert tel.to_dict()["dense_index"]["cache_hit"] is True


def test_resource_inventory_offline_safe():
    inv = resource_inventory()
    assert inv["cpu_logical"] is None or int(inv["cpu_logical"]) >= 1
    assert "gpu_count" in inv and "gpu_names" in inv
    assert isinstance(inv["gpu_names"], list)
