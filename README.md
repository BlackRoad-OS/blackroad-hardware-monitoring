<!-- BlackRoad SEO Enhanced -->

# ulackroad hardware monitoring

> Part of **[BlackRoad OS](https://blackroad.io)** — Sovereign Computing for Everyone

[![BlackRoad OS](https://img.shields.io/badge/BlackRoad-OS-ff1d6c?style=for-the-badge)](https://blackroad.io)
[![BlackRoad OS](https://img.shields.io/badge/Org-BlackRoad-OS-2979ff?style=for-the-badge)](https://github.com/BlackRoad-OS)
[![License](https://img.shields.io/badge/License-Proprietary-f5a623?style=for-the-badge)](LICENSE)

**ulackroad hardware monitoring** is part of the **BlackRoad OS** ecosystem — a sovereign, distributed operating system built on edge computing, local AI, and mesh networking by **BlackRoad OS, Inc.**

## About BlackRoad OS

BlackRoad OS is a sovereign computing platform that runs AI locally on your own hardware. No cloud dependencies. No API keys. No surveillance. Built by [BlackRoad OS, Inc.](https://github.com/BlackRoad-OS-Inc), a Delaware C-Corp founded in 2025.

### Key Features
- **Local AI** — Run LLMs on Raspberry Pi, Hailo-8, and commodity hardware
- **Mesh Networking** — WireGuard VPN, NATS pub/sub, peer-to-peer communication
- **Edge Computing** — 52 TOPS of AI acceleration across a Pi fleet
- **Self-Hosted Everything** — Git, DNS, storage, CI/CD, chat — all sovereign
- **Zero Cloud Dependencies** — Your data stays on your hardware

### The BlackRoad Ecosystem
| Organization | Focus |
|---|---|
| [BlackRoad OS](https://github.com/BlackRoad-OS) | Core platform and applications |
| [BlackRoad OS, Inc.](https://github.com/BlackRoad-OS-Inc) | Corporate and enterprise |
| [BlackRoad AI](https://github.com/BlackRoad-AI) | Artificial intelligence and ML |
| [BlackRoad Hardware](https://github.com/BlackRoad-Hardware) | Edge hardware and IoT |
| [BlackRoad Security](https://github.com/BlackRoad-Security) | Cybersecurity and auditing |
| [BlackRoad Quantum](https://github.com/BlackRoad-Quantum) | Quantum computing research |
| [BlackRoad Agents](https://github.com/BlackRoad-Agents) | Autonomous AI agents |
| [BlackRoad Network](https://github.com/BlackRoad-Network) | Mesh and distributed networking |
| [BlackRoad Education](https://github.com/BlackRoad-Education) | Learning and tutoring platforms |
| [BlackRoad Labs](https://github.com/BlackRoad-Labs) | Research and experiments |
| [BlackRoad Cloud](https://github.com/BlackRoad-Cloud) | Self-hosted cloud infrastructure |
| [BlackRoad Forge](https://github.com/BlackRoad-Forge) | Developer tools and utilities |

### Links
- **Website**: [blackroad.io](https://blackroad.io)
- **Documentation**: [docs.blackroad.io](https://docs.blackroad.io)
- **Chat**: [chat.blackroad.io](https://chat.blackroad.io)
- **Search**: [search.blackroad.io](https://search.blackroad.io)

---


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
