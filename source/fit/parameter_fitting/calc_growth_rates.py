import math


def growth_rate(doubling_time):
    return math.log(2) / doubling_time


def growth_rate_err(doubling_time, doubling_time_err):
    return math.log(2) * doubling_time_err / (doubling_time ** 2)


exp_doubling_time_sen = 84.92  # min
exp_doubling_time_sen_err = 1.75  # min
exp_doubling_time_res = 80.49  # min
exp_doubling_time_res_err = 1.21  # min

# 1 simulation step = 3 min because 10 updates correspond to one 30 min frame.
sen_doubling_time_sim = exp_doubling_time_sen / 3
sen_doubling_time_sim_err = exp_doubling_time_sen_err / 3
res_doubling_time_sim = exp_doubling_time_res / 3
res_doubling_time_sim_err = exp_doubling_time_res_err / 3

sen_30 = growth_rate(sen_doubling_time_sim)
sen_30_err = growth_rate_err(sen_doubling_time_sim, sen_doubling_time_sim_err)
res_30 = growth_rate(res_doubling_time_sim)
res_30_err = growth_rate_err(res_doubling_time_sim, res_doubling_time_sim_err)

print(f"Sensitive growth rate: {sen_30} +/- {sen_30_err}")
print(f"Resistant growth rate: {res_30} +/- {res_30_err}")
