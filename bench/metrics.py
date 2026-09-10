import json

def log_metrics(metrics, filename):
    with open(filename, 'a') as f:
        for metric in metrics:
            f.write(json.dumps(metric) + '\n')