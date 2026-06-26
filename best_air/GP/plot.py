import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(f'processed_uniform_low_step_5m_data.csv')

T_out = (data['zon_weaSta_reaWeaTDryBul_y'].values - 273.15)
time = data['time'].values

data = data.set_index('time')
x_time = data.index / 3600. / 24. -334
data['time'] = x_time 

plt.figure(figsize=(12, 6))
plt.plot(x_time, T_out, label='T_out')
plt.xlabel('Time (days)')
plt.ylabel('Temperature (°C)')
plt.title('Outdoor Temperature Over Time')
plt.grid()
plt.legend()
plt.show()
