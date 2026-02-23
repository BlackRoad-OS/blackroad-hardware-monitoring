# blackroad-hardware-monitoring

> Real-time hardware metrics collection, alerting, and anomaly detection — powered by SQLite.

## Features

- **Snapshot recording** — CPU, RAM, disk, temperature, network I/O, load averages
- **Threshold alerting** — configurable warning/critical levels per host and metric
- **Trend analysis** — linear-regression slope over any time window (no external deps)
- **Anomaly detection** — rolling z-score with configurable window and sensitivity
- **Health score** — weighted 0-100 composite score (cpu 30 % · ram 25 % · disk 20 % · temp 15 % · alerts 10 %)
- **Export** — JSON or CSV output for downstream ingestion
- **Rich terminal dashboard** — unicode bar charts and sparklines

## Quick Start

```bash
# Record a snapshot
python src/hardware_monitoring.py record \
  --host myserver --cpu 42.3 --ram 67.1 --disk 55.0 --temp 61.0

# Set alert thresholds
python src/hardware_monitoring.py threshold \
  --host myserver --metric cpu_percent --warning 75 --critical 92

# View dashboard
python src/hardware_monitoring.py dashboard --host myserver

# Check health score
python src/hardware_monitoring.py health-score --host myserver

# Show trend (last 24 h)
python src/hardware_monitoring.py trend --host myserver --metric cpu_percent

# Detect anomalies
python src/hardware_monitoring.py anomalies --host myserver --metric ram_percent

# Export last 12 h as CSV
python src/hardware_monitoring.py export --host myserver --hours 12 --format csv
```

## Requirements

- Python 3.9+ (stdlib only — no third-party packages required)

## Running Tests

```bash
pip install pytest
pytest tests/ -v --tb=short
```

## License

Proprietary — © BlackRoad OS, Inc. All rights reserved.
