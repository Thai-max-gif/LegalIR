"""F5: local benchmark/measurement protocol primitives."""
import pytest

from src.pipeline.kaggle_train import (
    NOMINAL_BUDGET_SECONDS,
    STRICT_GATE_SECONDS,
    StageTimingTelemetry,
    forecast_cold_total,
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


def _illustrative_jobs():
    # fix.md illustration (not a schedule): 5x700 fold + 700 disjoint + 875 final.
    return [{"updates": n, "sec_per_update": 0.0, "load_save_overhead": 0.0}
            for n in (700, 700, 700, 700, 700, 700, 875)]


def test_forecast_training_allowance_arithmetic():
    # 5075 updates; a 90-minute training allowance permits ~1.064 s/update
    # before any model load/save or validation overhead.
    jobs = _illustrative_jobs()
    assert sum(j["updates"] for j in jobs) == 5075
    out = forecast_cold_total(train_jobs=[{**j, "sec_per_update": 5400 / 5075} for j in jobs])
    assert out["training_seconds"] == pytest.approx(5400.0)
    assert out["total_updates"] == 5075
    # Any overhead on top breaks the allowance: forecast must say so.
    over = forecast_cold_total(
        train_jobs=[{**j, "sec_per_update": 5400 / 5075, "load_save_overhead": 60.0} for j in jobs]
    )
    assert over["training_seconds"] > 5400.0


def test_forecast_is_additive_not_multiplied():
    base = forecast_cold_total(setup_index_seconds=100.0, eval_queries=100, eval_qps=1.0)
    assert base["total_seconds"] == pytest.approx(200.0)
    halved_eval = forecast_cold_total(setup_index_seconds=100.0, eval_queries=100, eval_qps=2.0)
    assert halved_eval["total_seconds"] == pytest.approx(150.0)
    assert halved_eval["evaluation_seconds"] == pytest.approx(50.0)
    assert halved_eval["training_seconds"] == 0.0


def test_forecast_unknown_throughput_fails_closed():
    out = forecast_cold_total(eval_queries=8400, eval_qps=0)
    assert out["evaluation_seconds"] == float("inf")
    assert out["fits_nominal_270m"] is False
    assert out["fits_strict_300m"] is False


def test_forecast_budget_flags():
    assert NOMINAL_BUDGET_SECONDS == 270 * 60
    assert STRICT_GATE_SECONDS == 25200
    fits = forecast_cold_total(
        setup_index_seconds=100.0,
        train_jobs=[{"updates": 10, "sec_per_update": 1.0}],
    )
    assert fits["complete_measurements"] is True
    assert fits["fits_nominal_270m"] is True and fits["fits_strict_300m"] is True
    with pytest.raises(ValueError, match="negative"):
        forecast_cold_total(train_jobs=[{"updates": 10, "sec_per_update": -1.0}])


def test_forecast_empty_is_unknown_never_fitting():
    # The reported defect: no measurements yielded total 0 as passing evidence.
    out = forecast_cold_total()
    assert out["total_seconds"] == 0
    assert out["complete_measurements"] is False
    assert out["fits_nominal_270m"] is False
    assert out["fits_strict_300m"] is False
    # Training jobs without evaluation throughput are equally incomplete.
    out2 = forecast_cold_total(train_jobs=[{"updates": 100, "sec_per_update": 1.0}],
                               eval_queries=8400)
    assert out2["complete_measurements"] is False
    assert out2["fits_strict_300m"] is False


def test_forecast_cold_total_with_private_test_count():
    """Retime cold-run forecast with 2,080 private queries under measured A100 stage bounds."""
    train_jobs = [{"updates": n, "sec_per_update": 0.22, "load_save_overhead": 20.0}
                  for n in (700, 700, 700, 700, 700, 700, 875)]
    out = forecast_cold_total(
        setup_index_seconds=3338.0,
        static_retrieval_seconds=1800.0,
        train_jobs=train_jobs,
        eval_queries=8400,
        eval_qps=2.2,
        fusion_seconds=300.0,
        final_reload_public_seconds=995.5,
        delivery_seconds=390.0,
    )
    assert out["complete_measurements"] is True
    assert out["total_seconds"] <= NOMINAL_BUDGET_SECONDS
    assert out["total_seconds"] < STRICT_GATE_SECONDS
    assert out["fits_nominal_270m"] is True
    assert out["fits_strict_300m"] is True
