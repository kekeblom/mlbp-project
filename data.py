import csv
import numpy as np


def _parse_line(line):
    return list(map(int, line))

def load_table(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        headers = reader.__next__()
        lines = [_parse_line(line) for line in reader]
    table = np.array(lines)
    return headers, table[:, 0:-1], table[:, -1]

