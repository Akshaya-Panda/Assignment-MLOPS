"""Send realistic traffic to the API so the monitoring dashboards have data.

Patients are sampled from the cleaned dataset with small random perturbations; a share
of requests is deliberately invalid (to exercise 422 handling) and some are batches.
Standard library only.

usage: python scripts/generate_traffic.py [--url http://127.0.0.1:8000] [--duration 300]
                                          [--rps 5] [--invalid 0.05]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "processed" / "heart_cleveland_clean.csv"
INT_FIELDS = {"age", "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"}


def load_patients() -> list[dict]:
    patients = []
    with open(DATA, newline="") as fh:
        for row in csv.DictReader(fh):
            row.pop("target")
            patient = {}
            for key, value in row.items():
                if value == "":
                    continue  # optional fields (ca/thal) left out, as a client would
                patient[key] = int(float(value)) if key in INT_FIELDS else float(value)
            patients.append(patient)
    return patients


def jitter(patient: dict) -> dict:
    p = dict(patient)
    p["trestbps"] = round(min(max(p["trestbps"] + random.gauss(0, 4), 90), 200))
    p["chol"] = round(min(max(p["chol"] + random.gauss(0, 10), 120), 560))
    p["thalach"] = round(min(max(p["thalach"] + random.gauss(0, 5), 70), 205))
    return p


def invalid(patient: dict) -> dict:
    broken = dict(patient)
    choice = random.choice(["age", "cp", "missing", "extra"])
    if choice == "age":
        broken["age"] = 7
    elif choice == "cp":
        broken["cp"] = 9
    elif choice == "missing":
        broken.pop("thalach")
    else:
        broken["unexpected_field"] = 1
    return broken


def post(url: str, body: dict) -> int:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except (urllib.error.URLError, OSError):
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--duration", type=float, default=300, help="seconds")
    parser.add_argument("--rps", type=float, default=5, help="requests per second")
    parser.add_argument("--invalid", type=float, default=0.05, help="share of invalid requests")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    patients = load_patients()
    base = args.url.rstrip("/")
    statuses: Counter = Counter()
    end = time.monotonic() + args.duration
    sent = 0
    while time.monotonic() < end:
        roll = random.random()
        if roll < args.invalid:
            status = post(f"{base}/predict", invalid(random.choice(patients)))
        elif roll < args.invalid + 0.05:
            batch = [jitter(random.choice(patients)) for _ in range(random.randint(2, 8))]
            status = post(f"{base}/predict/batch", {"instances": batch})
        else:
            status = post(f"{base}/predict", jitter(random.choice(patients)))
        statuses[status] += 1
        sent += 1
        if sent % 100 == 0:
            print(f"{sent} requests sent, status counts: {dict(statuses)}", flush=True)
        time.sleep(random.expovariate(args.rps))
    print(f"done: {sent} requests, status counts: {dict(statuses)}")
    if statuses.get(0):
        raise SystemExit(f"{statuses[0]} requests could not reach {base}")


if __name__ == "__main__":
    main()
