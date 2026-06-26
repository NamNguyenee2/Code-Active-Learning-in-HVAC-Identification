import pandas as pd
import torch
import numpy as np

url = "http://127.0.0.1:80"
testcase = 'bestest_air'

Ns_tr, Ne_tr = 0, 10     
Ns_t, Ne_t = 200, 250  

norm_temp_constant = 30.0
norm_sup_constant = 40.0