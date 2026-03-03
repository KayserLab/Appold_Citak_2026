import numpy as np


def growth_rate(measurement):
    return np.log(2) / measurement


sen_30 = growth_rate(84.92*(1/3))
res_30 = growth_rate(80.49*(1/3))
print(sen_30, res_30)
