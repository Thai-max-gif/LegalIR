from src.utils.gate import gate_checks
from src.utils.runtime import report_status


def test_blocked_retrieval_propagates_not_pass():
    config = {"gate": {"vram_threshold_bytes": 14_000_000_000, "require_end_to_end": True, "require_approval": False}}
    run = {"checks": [], "training": {"optimizer_updates": 10, "parameter_changed": True}, "vram": {"peak_device_bytes": {"0": 1}}, "retrieval": {"status": "BLOCKED"}}
    checks = gate_checks(config, run)
    assert report_status(checks) == "BLOCKED"


def test_vram_excess_fails():
    config = {"gate": {"vram_threshold_bytes": 10, "require_end_to_end": False, "require_approval": False}}
    run = {"checks": [], "training": {"optimizer_updates": 5, "parameter_changed": True}, "vram": {"peak_device_bytes": {"0": 11}}, "retrieval": {"status": "PASS"}}
    assert report_status(gate_checks(config, run)) == "FAIL"
