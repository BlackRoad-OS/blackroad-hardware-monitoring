"""
Tests for src/hardware_monitoring.py
Run with: pytest tests/test_hardware_monitoring.py -v
"""

import json
import os
import sys
import tempfile

import pytest

# Allow import from sibling src/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hardware_monitoring import HardwareMonitor  # noqa: E402


@pytest.fixture
def mon(tmp_path):
    """Return a fresh HardwareMonitor backed by a temp SQLite file."""
    db = str(tmp_path / "test.db")
    monitor = HardwareMonitor(db_path=db)
    yield monitor
    monitor.close()


# ---------------------------------------------------------------------------
# 1. Record snapshot
# ---------------------------------------------------------------------------

def test_record_snapshot(mon):
    """Record one snapshot and verify it is persisted in the DB."""
    alerts = mon.record_snapshot(
        "host-a", cpu=45.0, ram=60.0, disk=30.0,
        temp=55.0, net_sent=1000, net_recv=2000,
        load1=1.5, load5=1.2, load15=1.0,
    )
    # No thresholds configured → no alerts
    assert alerts == []

    row = mon.conn.execute(
        "SELECT * FROM metric_snapshots WHERE hostname='host-a'"
    ).fetchone()
    assert row is not None
    assert abs(row["cpu_percent"] - 45.0) < 0.001
    assert abs(row["ram_percent"] - 60.0) < 0.001
    assert abs(row["disk_percent"] - 30.0) < 0.001
    assert abs(row["cpu_temp"] - 55.0) < 0.001
    assert row["net_bytes_sent"] == 1000
    assert row["net_bytes_recv"] == 2000


# ---------------------------------------------------------------------------
# 2. Threshold alert
# ---------------------------------------------------------------------------

def test_threshold_alert(mon):
    """Set a threshold and record a value exceeding the critical level."""
    mon.set_threshold("host-b", "cpu_percent", warning=70.0, critical=90.0)
    alerts = mon.record_snapshot(
        "host-b", cpu=95.0, ram=40.0, disk=20.0,
    )
    assert len(alerts) == 1
    alert = alerts[0]
    assert alert.level == "critical"
    assert alert.metric_name == "cpu_percent"
    assert alert.value == 95.0

    # Verify it was also persisted
    row = mon.conn.execute(
        "SELECT * FROM alert_events WHERE hostname='host-b'"
    ).fetchone()
    assert row is not None
    assert row["level"] == "critical"


# ---------------------------------------------------------------------------
# 3. Trend calculation
# ---------------------------------------------------------------------------

def test_trend_calculation(mon):
    """Insert 20 strictly increasing CPU values and verify slope > 0."""
    for i in range(20):
        mon.record_snapshot(
            "host-c", cpu=float(10 + i * 2), ram=50.0, disk=50.0,
        )
    trend = mon.get_trend("host-c", "cpu_percent", hours=24)
    assert trend["count"] == 20
    assert trend["slope"] > 0, f"Expected positive slope, got {trend['slope']}"
    assert trend["min"] < trend["max"]


# ---------------------------------------------------------------------------
# 4. Anomaly detection
# ---------------------------------------------------------------------------

def test_anomaly_detection(mon):
    """50 normal values (mean≈50) then one extreme (500) → anomaly detected."""
    for _ in range(50):
        mon.record_snapshot("host-d", cpu=50.0, ram=50.0, disk=50.0)
    # Insert extreme outlier
    mon.record_snapshot("host-d", cpu=500.0, ram=50.0, disk=50.0)

    anomalies = mon.detect_anomalies("host-d", "cpu_percent", window=50, z_threshold=2.5)
    assert len(anomalies) >= 1
    extreme = max(anomalies, key=lambda a: a.z_score)
    assert extreme.value == 500.0
    assert extreme.z_score > 2.5


# ---------------------------------------------------------------------------
# 5. Health score
# ---------------------------------------------------------------------------

def test_health_score(mon):
    """Recording high CPU and RAM usage should produce a health score below 50."""
    mon.record_snapshot(
        "host-e", cpu=97.0, ram=98.0, disk=80.0, temp=88.0,
    )
    # Add several unacknowledged alerts to pull the score further down
    mon.set_threshold("host-e", "cpu_percent", warning=70.0, critical=90.0)
    mon.set_threshold("host-e", "ram_percent", warning=75.0, critical=92.0)
    mon.record_snapshot("host-e", cpu=97.0, ram=98.0, disk=80.0, temp=88.0)

    score = mon.get_health_score("host-e")
    assert score < 50, f"Expected score < 50 but got {score}"
    assert 0 <= score <= 100


# ---------------------------------------------------------------------------
# 6. Export JSON
# ---------------------------------------------------------------------------

def test_export_json(mon):
    """Export recorded metrics as JSON and verify structure."""
    mon.record_snapshot("host-f", cpu=30.0, ram=40.0, disk=20.0, temp=45.0)
    mon.record_snapshot("host-f", cpu=35.0, ram=45.0, disk=22.0, temp=47.0)

    exported = mon.export_metrics("host-f", hours=24, fmt="json")
    data = json.loads(exported)

    assert isinstance(data, list)
    assert len(data) == 2

    required_keys = {
        "hostname", "timestamp", "cpu_percent", "ram_percent",
        "disk_percent", "cpu_temp",
    }
    for row in data:
        assert required_keys.issubset(row.keys()), (
            f"Missing keys in export: {required_keys - row.keys()}"
        )
        assert row["hostname"] == "host-f"
