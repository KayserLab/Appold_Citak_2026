import numpy as np


def growth_rate(measurement):
    return np.log(2) / measurement


sen_30 = growth_rate(84.92*(1/3))  # (1/0.18) sim_time  0.18s
res_30 = growth_rate(80.49*(1/3))  # sim_time
print(sen_30, res_30)
