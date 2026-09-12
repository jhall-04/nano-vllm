import argparse
import glob
import json
import os
import re
import statistics

LOG_NAME_RE = re.compile(r"load_test_(\d+)qps\.log$")


def percentile(sorted_values, pct):
    if not sorted_values:
        return None
    k = (len(sorted_values) - 1) * (pct / 100)
    lo, hi = int(k), min(int(k) + 1, len(sorted_values) - 1)
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo)


def summarize(filepath):
    response_times = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            response_times.append(json.loads(line)["response_time"])

    if not response_times:
        return None

    response_times.sort()
    return {
        "count": len(response_times),
        "min": response_times[0],
        "max": response_times[-1],
        "mean": statistics.mean(response_times),
        "p50": percentile(response_times, 50),
        "p95": percentile(response_times, 95),
        "p99": percentile(response_times, 99),
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize load test logs by version and QPS.")
    parser.add_argument("--logs-dir", default="logs", help="Directory containing logs-v*/ folders")
    args = parser.parse_args()

    rows = []
    for version_dir in sorted(glob.glob(os.path.join(args.logs_dir, "logs-v*"))):
        version = os.path.basename(version_dir)
        for filepath in sorted(glob.glob(os.path.join(version_dir, "load_test_*qps.log"))):
            match = LOG_NAME_RE.search(filepath)
            if not match:
                continue
            qps = int(match.group(1))
            stats = summarize(filepath)
            if stats is None:
                continue
            rows.append({"version": version, "qps": qps, **stats})

    rows.sort(key=lambda r: (r["version"], r["qps"]))

    header = f"{'version':<10}{'qps':>5}{'count':>8}{'min':>10}{'max':>10}{'mean':>10}{'p50':>10}{'p95':>10}{'p99':>10}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['version']:<10}{row['qps']:>5}{row['count']:>8}"
            f"{row['min']:>10.3f}{row['max']:>10.3f}{row['mean']:>10.3f}"
            f"{row['p50']:>10.3f}{row['p95']:>10.3f}{row['p99']:>10.3f}"
        )


if __name__ == "__main__":
    main()
