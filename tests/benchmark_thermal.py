"""CPU-only thermal guards for benchmarks already running inside MetalQ."""

import json
import time
from pathlib import Path

THERMAL_FILE = Path.home() / ".metalq" / "thermal-state"


def thermal_snapshot():
    """Read MetalQ's live ProcessInfo helper; fail rather than assume nominal."""
    record = json.loads(THERMAL_FILE.read_text())
    now = time.time()
    if not 0 <= now - record["ts"] <= 5:
        raise RuntimeError("MetalQ thermal helper is stale")
    return {"observed_at": now, "state": int(record["state"])}


def wait_for_nominal(seconds, timeout=300):
    """Require continuous nominal samples before each arm, outside timed work."""
    started = time.monotonic()
    nominal_since = None
    samples = []
    while True:
        now = time.monotonic()
        sample = thermal_snapshot()
        samples.append(sample)
        if sample["state"] == 0:
            if nominal_since is None:
                nominal_since = now
            if now - nominal_since >= seconds:
                return {"wait_seconds": now - started, "samples": samples}
        else:
            nominal_since = None
        if now - started >= timeout:
            raise RuntimeError("Nominal thermal hold timed out; no measurement started")
        time.sleep(
            min(1, max(0.01, seconds - (now - nominal_since)))
            if nominal_since is not None
            else 1
        )
