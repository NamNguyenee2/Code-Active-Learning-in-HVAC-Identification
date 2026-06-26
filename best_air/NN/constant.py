



step = 5
step_per_day = 1440//step

ramp = 0.2
Ns_tr = 0
Ne_tr = 10
start_test = 5
end_test = 6
Ns_t = int(start_test*step_per_day)
Ne_t = int(end_test*step_per_day)

num_count = int(step_per_day*1)

hidden_dim = 8
epoch_ini = 500
epoch_onl = 100
loop = 5

N_bootstrap = 10
dropout_p = 0.1
lr = 0.0005

number_of_data = 10
count_data = 1