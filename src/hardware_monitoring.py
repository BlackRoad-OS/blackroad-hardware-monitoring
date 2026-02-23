"""
BlackRoad Hardware Monitoring — src/hardware_monitoring.py
Collects, stores, analyses, and alerts on hardware metrics via SQLite.
"""

import argparse
import csv
import io
import json
import math
import os
import sqlite3
import statistics
import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# ANSI colours
# ---------------------------------------------------------------------------
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MetricSnapshot:
    id: Optional[int]
    hostname: str
    timestamp: str
    cpu_percent: float
    ram_percent: float
    disk_percent: float
    cpu_temp: float
    net_bytes_sent: int
    net_bytes_recv: int
    load_avg_1m: float
    load_avg_5m: float
    load_avg_15m: float


@dataclass
class AlertThreshold:
    id: Optional[int]
    hostname: str
    metric_name: str
    warning_level: float
    critical_level: float
    enabled: bool = True


@dataclass
class AlertEvent:
    id: Optional[int]
    hostname: str
    metric_name: str
    value: float
    level: str          # 'warning' | 'critical'
    message: str
    timestamp: str
    acknowledged: bool = False


@dataclass
class AnomalyEvent:
    id: Optional[int]
    hostname: str
    metric_name: str
    value: float
    z_score: float
    timestamp: str
    window_size: int


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class HardwareMonitor:
    """SQLite-backed hardware metrics collector, analyser, and alerter."""

    METRIC_COLUMNS = [
        "cpu_percent", "ram_percent", "disk_percent",
        "cpu_temp", "load_avg_1m", "load_avg_5m", "load_avg_15m",
    ]

    def __init__(self, db_path: str = "hardware_monitoring.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.init_db()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def init_db(self) -> None:
        """Create the four core tables if they do not already exist."""
        cur = self.conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS metric_snapshots (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                hostname        TEXT    NOT NULL,
                timestamp       TEXT    NOT NULL,
                cpu_percent     REAL    NOT NULL,
                ram_percent     REAL    NOT NULL,
                disk_percent    REAL    NOT NULL,
                cpu_temp        REAL    NOT NULL DEFAULT 0,
                net_bytes_sent  INTEGER NOT NULL DEFAULT 0,
                net_bytes_recv  INTEGER NOT NULL DEFAULT 0,
                load_avg_1m     REAL    NOT NULL DEFAULT 0,
                load_avg_5m     REAL    NOT NULL DEFAULT 0,
                load_avg_15m    REAL    NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS alert_thresholds (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                hostname        TEXT    NOT NULL,
                metric_name     TEXT    NOT NULL,
                warning_level   REAL    NOT NULL,
                critical_level  REAL    NOT NULL,
                enabled         INTEGER NOT NULL DEFAULT 1,
                UNIQUE(hostname, metric_name)
            );

            CREATE TABLE IF NOT EXISTS alert_events (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                hostname        TEXT    NOT NULL,
                metric_name     TEXT    NOT NULL,
                value           REAL    NOT NULL,
                level           TEXT    NOT NULL,
                message         TEXT    NOT NULL,
                timestamp       TEXT    NOT NULL,
                acknowledged    INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS anomaly_events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                hostname    TEXT    NOT NULL,
                metric_name TEXT    NOT NULL,
                value       REAL    NOT NULL,
                z_score     REAL    NOT NULL,
                timestamp   TEXT    NOT NULL,
                window_size INTEGER NOT NULL
            );
        """)
        self.conn.commit()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_snapshot(
        self,
        hostname: str,
        cpu: float,
        ram: float,
        disk: float,
        temp: float = 0.0,
        net_sent: int = 0,
        net_recv: int = 0,
        load1: float = 0.0,
        load5: float = 0.0,
        load15: float = 0.0,
    ) -> List[AlertEvent]:
        """Insert a new snapshot and evaluate alert thresholds."""
        ts = datetime.now(timezone.utc).isoformat()
        cur = self.conn.cursor()
        cur.execute(
            """INSERT INTO metric_snapshots
               (hostname, timestamp, cpu_percent, ram_percent, disk_percent,
                cpu_temp, net_bytes_sent, net_bytes_recv,
                load_avg_1m, load_avg_5m, load_avg_15m)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (hostname, ts, cpu, ram, disk, temp, net_sent, net_recv, load1, load5, load15),
        )
        self.conn.commit()
        snap = MetricSnapshot(
            id=cur.lastrowid,
            hostname=hostname,
            timestamp=ts,
            cpu_percent=cpu,
            ram_percent=ram,
            disk_percent=disk,
            cpu_temp=temp,
            net_bytes_sent=net_sent,
            net_bytes_recv=net_recv,
            load_avg_1m=load1,
            load_avg_5m=load5,
            load_avg_15m=load15,
        )
        return self.check_thresholds(hostname, snap)

    # ------------------------------------------------------------------
    # Thresholds
    # ------------------------------------------------------------------

    def set_threshold(
        self,
        hostname: str,
        metric_name: str,
        warning: float,
        critical: float,
    ) -> None:
        """Upsert an alert threshold for a host+metric combination."""
        self.conn.execute(
            """INSERT INTO alert_thresholds (hostname, metric_name, warning_level, critical_level)
               VALUES (?,?,?,?)
               ON CONFLICT(hostname, metric_name)
               DO UPDATE SET warning_level=excluded.warning_level,
                             critical_level=excluded.critical_level,
                             enabled=1""",
            (hostname, metric_name, warning, critical),
        )
        self.conn.commit()

    def check_thresholds(
        self, hostname: str, snapshot: MetricSnapshot
    ) -> List[AlertEvent]:
        """Compare snapshot values to thresholds; persist and return fired alerts."""
        rows = self.conn.execute(
            "SELECT * FROM alert_thresholds WHERE hostname=? AND enabled=1",
            (hostname,),
        ).fetchall()

        fired: List[AlertEvent] = []
        ts = datetime.now(timezone.utc).isoformat()

        for row in rows:
            metric = row["metric_name"]
            value = getattr(snapshot, metric, None)
            if value is None:
                continue

            level: Optional[str] = None
            if value >= row["critical_level"]:
                level = "critical"
            elif value >= row["warning_level"]:
                level = "warning"

            if level:
                msg = (
                    f"{metric} is {value:.1f} on {hostname} "
                    f"(threshold {level}: {row[level + '_level']:.1f})"
                )
                cur = self.conn.execute(
                    """INSERT INTO alert_events
                       (hostname, metric_name, value, level, message, timestamp)
                       VALUES (?,?,?,?,?,?)""",
                    (hostname, metric, value, level, msg, ts),
                )
                self.conn.commit()
                fired.append(
                    AlertEvent(
                        id=cur.lastrowid,
                        hostname=hostname,
                        metric_name=metric,
                        value=value,
                        level=level,
                        message=msg,
                        timestamp=ts,
                    )
                )
        return fired

    # ------------------------------------------------------------------
    # Trend analysis
    # ------------------------------------------------------------------

    def get_trend(
        self, hostname: str, metric_name: str, hours: int = 24
    ) -> Dict:
        """Compute linear-regression slope + descriptive stats for a metric."""
        since = (
            datetime.now(timezone.utc) - timedelta(hours=hours)
        ).isoformat()
        rows = self.conn.execute(
            f"SELECT timestamp, {metric_name} AS val FROM metric_snapshots "
            f"WHERE hostname=? AND timestamp>=? ORDER BY timestamp ASC",
            (hostname, since),
        ).fetchall()

        if not rows:
            return {"slope": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0, "stddev": 0.0, "count": 0}

        values = [r["val"] for r in rows]
        n = len(values)
        xs = list(range(n))

        mean_x = sum(xs) / n
        mean_y = sum(values) / n
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values))
        den = sum((x - mean_x) ** 2 for x in xs)
        slope = num / den if den != 0 else 0.0

        stddev = statistics.stdev(values) if n > 1 else 0.0

        return {
            "slope": round(slope, 6),
            "mean": round(mean_y, 2),
            "min": round(min(values), 2),
            "max": round(max(values), 2),
            "stddev": round(stddev, 2),
            "count": n,
        }

    # ------------------------------------------------------------------
    # Anomaly detection
    # ------------------------------------------------------------------

    def detect_anomalies(
        self,
        hostname: str,
        metric_name: str,
        window: int = 50,
        z_threshold: float = 2.5,
    ) -> List[AnomalyEvent]:
        """Rolling z-score anomaly detection; persists and returns events."""
        rows = self.conn.execute(
            f"SELECT id, timestamp, {metric_name} AS val FROM metric_snapshots "
            f"WHERE hostname=? ORDER BY timestamp ASC",
            (hostname,),
        ).fetchall()

        detected: List[AnomalyEvent] = []
        buf: deque = deque(maxlen=window)

        for row in rows:
            val = row["val"]
            if len(buf) >= 2:
                mean = sum(buf) / len(buf)
                std = statistics.stdev(buf) if len(buf) > 1 else 0.0
                if std > 0:
                    z = abs(val - mean) / std
                    if z >= z_threshold:
                        cur = self.conn.execute(
                            """INSERT OR IGNORE INTO anomaly_events
                               (hostname, metric_name, value, z_score, timestamp, window_size)
                               VALUES (?,?,?,?,?,?)""",
                            (hostname, metric_name, val, round(z, 3), row["timestamp"], len(buf)),
                        )
                        self.conn.commit()
                        detected.append(
                            AnomalyEvent(
                                id=cur.lastrowid,
                                hostname=hostname,
                                metric_name=metric_name,
                                value=val,
                                z_score=round(z, 3),
                                timestamp=row["timestamp"],
                                window_size=len(buf),
                            )
                        )
            buf.append(val)
        return detected

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    def get_dashboard(self, hostname: str) -> Dict:
        """Return latest snapshot, active alerts, anomaly count, trend slopes."""
        snap_row = self.conn.execute(
            "SELECT * FROM metric_snapshots WHERE hostname=? ORDER BY timestamp DESC LIMIT 1",
            (hostname,),
        ).fetchone()

        latest = dict(snap_row) if snap_row else {}

        active_alerts = [
            dict(r)
            for r in self.conn.execute(
                "SELECT * FROM alert_events WHERE hostname=? AND acknowledged=0 ORDER BY timestamp DESC LIMIT 20",
                (hostname,),
            ).fetchall()
        ]

        anomaly_count = self.conn.execute(
            "SELECT COUNT(*) AS cnt FROM anomaly_events WHERE hostname=?",
            (hostname,),
        ).fetchone()["cnt"]

        trends = {m: self.get_trend(hostname, m) for m in self.METRIC_COLUMNS}

        return {
            "hostname": hostname,
            "latest_snapshot": latest,
            "active_alerts": active_alerts,
            "anomaly_count": anomaly_count,
            "trends": trends,
        }

    # ------------------------------------------------------------------
    # Alert management
    # ------------------------------------------------------------------

    def acknowledge_alert(self, alert_id: int) -> bool:
        """Mark an alert as acknowledged. Returns True if a row was updated."""
        cur = self.conn.execute(
            "UPDATE alert_events SET acknowledged=1 WHERE id=?", (alert_id,)
        )
        self.conn.commit()
        return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_metrics(
        self, hostname: str, hours: int = 24, fmt: str = "json"
    ) -> str:
        """Export snapshots as JSON or CSV string."""
        since = (
            datetime.now(timezone.utc) - timedelta(hours=hours)
        ).isoformat()
        rows = self.conn.execute(
            "SELECT * FROM metric_snapshots WHERE hostname=? AND timestamp>=? ORDER BY timestamp ASC",
            (hostname, since),
        ).fetchall()

        data = [dict(r) for r in rows]

        if fmt == "csv":
            if not data:
                return ""
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=list(data[0].keys()))
            writer.writeheader()
            writer.writerows(data)
            return buf.getvalue()

        return json.dumps(data, indent=2)

    # ------------------------------------------------------------------
    # Health score
    # ------------------------------------------------------------------

    def get_health_score(self, hostname: str) -> float:
        """
        Weighted 0-100 score.
        cpu 30% | ram 25% | disk 20% | temp 15% | unacked-alerts 10%
        """
        snap_row = self.conn.execute(
            "SELECT * FROM metric_snapshots WHERE hostname=? ORDER BY timestamp DESC LIMIT 1",
            (hostname,),
        ).fetchone()

        if not snap_row:
            return 100.0

        snap = dict(snap_row)

        def _penalty(value: float, warn: float, crit: float) -> float:
            """Returns 0 (good) to 1 (worst) penalty fraction."""
            if value >= crit:
                return 1.0
            if value >= warn:
                return 0.5 + 0.5 * (value - warn) / max(crit - warn, 1e-9)
            return value / max(warn, 1e-9) * 0.5

        cpu_p = _penalty(snap["cpu_percent"], 70, 90)
        ram_p = _penalty(snap["ram_percent"], 75, 92)
        disk_p = _penalty(snap["disk_percent"], 80, 95)
        temp = snap.get("cpu_temp") or 0.0
        temp_p = _penalty(temp, 70, 90) if temp > 0 else 0.0

        unacked = self.conn.execute(
            "SELECT COUNT(*) AS cnt FROM alert_events WHERE hostname=? AND acknowledged=0",
            (hostname,),
        ).fetchone()["cnt"]
        alert_p = min(unacked / 10.0, 1.0)

        weighted_penalty = (
            cpu_p * 0.30
            + ram_p * 0.25
            + disk_p * 0.20
            + temp_p * 0.15
            + alert_p * 0.10
        )
        return round((1.0 - weighted_penalty) * 100, 1)

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def close(self) -> None:
        self.conn.close()


# ---------------------------------------------------------------------------
# Terminal helpers
# ---------------------------------------------------------------------------

BLOCKS = " ▁▂▃▄▅▆▇█"


def _bar(value: float, width: int = 20, low_good: bool = True) -> str:
    """Return a unicode block progress bar with colour coding."""
    pct = max(0.0, min(100.0, value))
    filled = int(pct / 100 * width)
    bar = BLOCKS[-1] * filled + BLOCKS[0] * (width - filled)
    if low_good:
        colour = GREEN if pct < 60 else (YELLOW if pct < 85 else RED)
    else:
        colour = CYAN
    return f"{colour}[{bar}]{RESET} {pct:5.1f}%"


def _sparkline(values: List[float]) -> str:
    """Render a compact sparkline from a list of floats."""
    if not values:
        return ""
    lo, hi = min(values), max(values)
    span = hi - lo or 1.0
    return "".join(BLOCKS[int((v - lo) / span * (len(BLOCKS) - 1))] for v in values)


def _print_dashboard(dash: Dict) -> None:
    host = dash["hostname"]
    snap = dash["latest_snapshot"]
    print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}")
    print(f"{BOLD}{BLUE}  Hardware Monitor — {host}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*60}{RESET}")

    if not snap:
        print(f"{YELLOW}  No data recorded yet for {host}.{RESET}\n")
        return

    metrics = [
        ("CPU Usage", snap.get("cpu_percent", 0)),
        ("RAM Usage", snap.get("ram_percent", 0)),
        ("Disk Usage", snap.get("disk_percent", 0)),
        ("CPU Temp °C", snap.get("cpu_temp", 0)),
        ("Load 1m",  snap.get("load_avg_1m", 0)),
    ]
    for label, val in metrics:
        low_good = label != "Load 1m"
        print(f"  {label:<14} {_bar(val, low_good=low_good)}")

    print(f"\n{BOLD}  Alerts (unacked): {len(dash['active_alerts'])}{RESET}")
    for alert in dash["active_alerts"][:5]:
        colour = RED if alert["level"] == "critical" else YELLOW
        print(f"    {colour}[{alert['level'].upper()}]{RESET} {alert['message']}")

    print(f"\n{BOLD}  Anomalies detected (all-time): {dash['anomaly_count']}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*60}{RESET}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hardware_monitoring",
        description="BlackRoad Hardware Monitoring CLI",
    )
    p.add_argument("--db", default="hardware_monitoring.db", help="SQLite database path")
    sub = p.add_subparsers(dest="cmd", required=True)

    # record
    r = sub.add_parser("record", help="Record a metric snapshot")
    r.add_argument("--host", required=True)
    r.add_argument("--cpu", type=float, required=True)
    r.add_argument("--ram", type=float, required=True)
    r.add_argument("--disk", type=float, required=True)
    r.add_argument("--temp", type=float, default=0.0)
    r.add_argument("--net-sent", type=int, default=0)
    r.add_argument("--net-recv", type=int, default=0)
    r.add_argument("--load1", type=float, default=0.0)
    r.add_argument("--load5", type=float, default=0.0)
    r.add_argument("--load15", type=float, default=0.0)

    # dashboard
    d = sub.add_parser("dashboard", help="Show live dashboard for a host")
    d.add_argument("--host", required=True)

    # threshold
    t = sub.add_parser("threshold", help="Set an alert threshold")
    t.add_argument("--host", required=True)
    t.add_argument("--metric", required=True)
    t.add_argument("--warning", type=float, required=True)
    t.add_argument("--critical", type=float, required=True)

    # alerts
    al = sub.add_parser("alerts", help="List unacknowledged alerts")
    al.add_argument("--host", required=True)
    al.add_argument("--ack", type=int, default=None, help="Acknowledge alert by ID")

    # trend
    tr = sub.add_parser("trend", help="Show trend for a metric")
    tr.add_argument("--host", required=True)
    tr.add_argument("--metric", required=True)
    tr.add_argument("--hours", type=int, default=24)

    # anomalies
    an = sub.add_parser("anomalies", help="Detect anomalies")
    an.add_argument("--host", required=True)
    an.add_argument("--metric", required=True)
    an.add_argument("--window", type=int, default=50)
    an.add_argument("--z", type=float, default=2.5)

    # export
    ex = sub.add_parser("export", help="Export metrics as JSON or CSV")
    ex.add_argument("--host", required=True)
    ex.add_argument("--hours", type=int, default=24)
    ex.add_argument("--format", choices=["json", "csv"], default="json")

    # health-score
    hs = sub.add_parser("health-score", help="Compute composite health score")
    hs.add_argument("--host", required=True)

    return p


def main(argv=None) -> None:  # noqa: C901
    parser = _build_parser()
    args = parser.parse_args(argv)
    mon = HardwareMonitor(db_path=args.db)

    try:
        if args.cmd == "record":
            alerts = mon.record_snapshot(
                args.host, args.cpu, args.ram, args.disk, args.temp,
                args.net_sent, args.net_recv, args.load1, args.load5, args.load15,
            )
            print(f"{GREEN}✓ Snapshot recorded for {args.host}{RESET}")
            for a in alerts:
                col = RED if a.level == "critical" else YELLOW
                print(f"  {col}[{a.level.upper()}]{RESET} {a.message}")

        elif args.cmd == "dashboard":
            _print_dashboard(mon.get_dashboard(args.host))

        elif args.cmd == "threshold":
            mon.set_threshold(args.host, args.metric, args.warning, args.critical)
            print(
                f"{GREEN}✓ Threshold set: {args.metric} warn={args.warning} crit={args.critical}{RESET}"
            )

        elif args.cmd == "alerts":
            if args.ack is not None:
                ok = mon.acknowledge_alert(args.ack)
                print(f"{GREEN if ok else RED}✓ Alert {args.ack} {'acknowledged' if ok else 'not found'}{RESET}")
            else:
                dash = mon.get_dashboard(args.host)
                for a in dash["active_alerts"]:
                    col = RED if a["level"] == "critical" else YELLOW
                    print(f"  [{a['id']}] {col}[{a['level'].upper()}]{RESET} {a['message']}")

        elif args.cmd == "trend":
            t = mon.get_trend(args.host, args.metric, args.hours)
            print(f"\n{BOLD}Trend: {args.metric} on {args.host} (last {args.hours}h){RESET}")
            direction = (
                f"{RED}↑ rising{RESET}" if t["slope"] > 0
                else (f"{GREEN}↓ falling{RESET}" if t["slope"] < 0 else "→ flat")
            )
            print(f"  Slope  : {t['slope']:+.4f}  {direction}")
            print(f"  Mean   : {t['mean']}")
            print(f"  Min    : {t['min']}   Max: {t['max']}")
            print(f"  StdDev : {t['stddev']}   Samples: {t['count']}\n")

        elif args.cmd == "anomalies":
            events = mon.detect_anomalies(args.host, args.metric, args.window, args.z)
            print(f"{MAGENTA}Anomalies detected: {len(events)}{RESET}")
            for ev in events:
                print(f"  z={ev.z_score:.2f}  val={ev.value}  @ {ev.timestamp}")

        elif args.cmd == "export":
            print(mon.export_metrics(args.host, args.hours, args.format))

        elif args.cmd == "health-score":
            score = mon.get_health_score(args.host)
            colour = GREEN if score >= 75 else (YELLOW if score >= 50 else RED)
            print(f"\n{BOLD}Health Score — {args.host}{RESET}")
            bar = _bar(score, low_good=False)
            print(f"  {bar}  {colour}{BOLD}{score}/100{RESET}\n")

    finally:
        mon.close()


if __name__ == "__main__":
    main()
