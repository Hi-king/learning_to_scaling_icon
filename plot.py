import pylab
import sys
import argparse
import numpy

parser = argparse.ArgumentParser()
parser.add_argument("window_size", type=int)
parser.add_argument("--skip", type=int, default=0)
args = parser.parse_args()

values = []
window = [0 for i in range(args.window_size)]
sum_value = 0
for i, line in enumerate(sys.stdin):
    value = float(line.rstrip())
    window.append(value)
    sum_value += value
    sum_value -= window.pop(0)
    if i > args.skip:
        values.append(float(sum_value) / args.window_size)

pylab.plot(values)
pylab.show()
