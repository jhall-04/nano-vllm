import argparse

def poisson_interval(qps):
    return 1.0 / qps

def main():
    parser = argparse.ArgumentParser(description="Load generator for inference benchmarking.")

    parser.add_argument("--qps", type=int, default=10, help="Queries per second to generate")
    parser.add_argument("--duration", type=int, default=60, help="Duration of the load test in seconds")

    args = parser.parse_args()

if __name__ == "__main__":
    main()