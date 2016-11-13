import csv
import numpy as np


def _parse_line(line):
    return list(map(int, line))

def _read_table(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        headers = reader.__next__()
        lines = [_parse_line(line) for line in reader]
    return headers, np.array(lines)

def load_training_data(filename):
    headers, table = _read_table(filename)
    return headers, table[:, 0:-1], table[:, -1]

def load_test_data(filename):
    headers, table = _read_table(filename)
    return headers, table

def load_test_labels(filename):
    headers, labels = _read_table(filename)
    return headers, labels[:, 1]
