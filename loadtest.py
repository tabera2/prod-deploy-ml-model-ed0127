"""A minimal load test: fire N concurrent predictions, report latency tail."""
import time
import statistics
import concurrent.futures as cf

import httpx

URL = "http://localhost:8000/predict"
SAMPLE = {"features": [5.1, 3.5, 1.4, 0.2]}
N_REQUESTS = 500
CONCURRENCY = 20


def one_call() -> float:
    start = time.perf_counter()
    r = httpx.post(URL, json=SAMPLE, timeout=5.0)
    r.raise_for_status()
    return time.perf_counter() - start


def main() -> None:
    with cf.ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        latencies = list(pool.map(lambda _: one_call(), range(N_REQUESTS)))
    latencies.sort()
    p50 = statistics.median(latencies)
    p99 = latencies[int(len(latencies) * 0.99) - 1]
    print(f"requests={N_REQUESTS} p50={p50*1000:.1f}ms p99={p99*1000:.1f}ms")


if __name__ == "__main__":
    main()
