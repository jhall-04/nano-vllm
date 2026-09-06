import json

def log_metrics(metrics, filename):
    for metric in metrics:
        with open(filename, 'a') as f:
            f.write(json.dumps(metric) + '\n')