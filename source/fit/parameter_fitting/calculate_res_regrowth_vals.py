import numpy as np
import matplotlib.pyplot as plt

def f(x):
    x = x * 13.76 * 8.648
    y = (x + 8.80313863285115*15.828348487292509) / 15.828348487292509  # taken from Fig. 2 in the paper
    return y*2  # times 2 to convert from hours to frames (30 min per frame)

print(f(3), f(7), f(11), f(15))
