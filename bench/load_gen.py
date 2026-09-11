import argparse
import json
import os
from metrics import log_metrics
from datetime import datetime, timedelta
import asyncio
import aiohttp

async def send_request(url, payload):
    start_time = datetime.now()
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
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
        payload = {
            "prompt": "Hello, world!",
            "decode_params": {},
        }
        url = "http://localhost:8000/generate"
        print("Creating request task at", datetime.now().isoformat())
        task = asyncio.create_task(send_request(url, payload))
        tasks.append(task)
    logs = await asyncio.gather(*tasks)

    return logs

def golden_test(golden_prompt_file="tests/golden_with_responses.json"):
    logs = []
    if not os.path.exists(golden_prompt_file):
        print(f"Golden prompt file {golden_prompt_file} does not exist.")
        return []
    with open(golden_prompt_file, 'r') as f:
        golden_prompts = json.load(f)
    decode_params = golden_prompts.get("decoding", {})
    for prompt in golden_prompts["prompts"]:
        max_new_tokens = prompt.get("max_new_tokens", 512)
        decode_params["max_new_tokens"] = max_new_tokens
        response = asyncio.run(send_request("http://localhost:8000/generate", {"prompt": prompt["text"], "decode_params": decode_params}))
        if prompt.get("expected_response") and response["response"].get("message") != prompt["expected_response"]:
            match = False
        else:
            match = True
        logs.append({"prompt": prompt["text"], "result": response, "match": match})
    return logs

def main():
    parser = argparse.ArgumentParser(description="Load generator for inference benchmarking.")

    parser.add_argument("--test", type=str, default="load_test", help="Name of the test")
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

    if args.test == "load_test":
        logs = asyncio.run(simulate_load(args.qps, args.duration))
    elif args.test == "golden_test":
        logs = golden_test()
    else:
        print(f"Unknown test: {args.test}")
        return

    log_metrics(logs, args.output)



if __name__ == "__main__":
    main()