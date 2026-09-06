import argparse
import os
from metrics import log_metrics
from datetime import datetime, timedelta
import asyncio
import aiohttp

async def send_request(url, params):
    start_time = datetime.now()
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            response_data = await response.json()
            end_time = datetime.now()
            return {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "response_time": (end_time - start_time).total_seconds(),
                "status_code": response.status,
                "response": response_data
            }

async def simulate_load(qps, duration):
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=duration)
    logs = []
    tasks = []
    while datetime.now() < end_time:
        last_interval = 1 / qps
        await asyncio.sleep(last_interval)
        params = {"prompt": "Hello, world!"}
        url = "http://localhost:8000/generate"
        task = asyncio.create_task(send_request(url, params))
        tasks.append(task)
    logs = await asyncio.gather(*tasks)

    return logs

def main():
    parser = argparse.ArgumentParser(description="Load generator for inference benchmarking.")

    parser.add_argument("--qps", type=int, default=10, help="Queries per second to generate")
    parser.add_argument("--duration", type=int, default=60, help="Duration of the load test in seconds")
    parser.add_argument("--output", type=str, default="metrics.log", help="Output file for metrics")

    args = parser.parse_args()

    if args.qps <= 0:
        print("QPS must be a positive integer.")
        return

    if args.duration <= 0:
        print("Duration must be a positive integer.")
        return

    if os.path.exists(args.output):
        os.remove(args.output)

    with open(args.output, 'w') as f:
        f.write("")

    logs = asyncio.run(simulate_load(args.qps, args.duration))
    log_metrics(logs, args.output)



if __name__ == "__main__":
    main()