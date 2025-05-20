import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import pysindy as ps
from sklearn.metrics import mean_squared_error
import math
np.math = math


def name_constant(input_features):
    return "1"

def name_quadratic(input_features):
    return f"{input_features[0]}^2"

def simulate_bundle_full(T_end, max_steps, num=6, r=0.01, s=0.074, Ntot=100000):
    m = np.full((num, max_steps + 1), 1, dtype=np.int32)
    T = np.zeros(max_steps + 1, dtype=float)

    for step in range(max_steps):
        fil = np.random.randint(0, num)

        k1 = s * m[fil, step]
        k2 = r * Ntot
        k0 = k1 + k2
        T[step + 1] = T[step] + np.random.exponential(1 / k0)

        if np.random.rand() < k1 / k0 and m[fil, step] > 1:
            cut = np.random.randint(1, m[fil, step])
            new_len = m[fil, step] - cut
        else:
            new_len = m[fil, step] + 1

        m[:, step + 1] = m[:, step]
        m[fil, step + 1] = new_len

        if T[step + 1] >= T_end:
            break

    idx = step + 1
    return T[: idx + 1], m[:, : idx + 1]

T_end, max_steps = 30.0, 3000000
num_traj, num_points = 5000, 1000
t_uniform = np.linspace(0, T_end, num_points)

L_all = np.zeros((num_traj, num_points))
for k in range(num_traj):
    T_raw, m_raw = simulate_bundle_full(T_end, max_steps)
    L_max = m_raw.max(axis=0)
    L_interp = interp1d(T_raw, L_max,
                        kind='linear', fill_value='extrapolate')
    L_all[k] = L_interp(t_uniform)
L_max_avg = L_all.mean(axis=0)
L_max_smooth = savgol_filter(L_max_avg, window_length=51, polyorder=3)

L_avg = L_all.mean(axis=0)

L_avg_smooth = savgol_filter(L_avg, window_length=51, polyorder=3)
dt = t_uniform[1] - t_uniform[0]
dLmax_dt = np.gradient(L_max_smooth, dt)
dL_dt = np.gradient(L_avg_smooth, dt)

X = L_max_smooth.reshape(-1, 1)
library = ps.PolynomialLibrary(degree=2, include_bias=True)
# library_functions = [lambda x: 1, lambda x: x**2]
# function_names = [ name_constant, name_quadratic ]
# library = ps.CustomLibrary(library_functions=library_functions)
model = ps.SINDy(feature_library=library,
                 optimizer=ps.STLSQ(threshold=1e-6, alpha=1e-3))
model.fit(X, t=dt, x_dot=dLmax_dt.reshape(-1, 1))
pred_max = model.simulate([L_max_smooth[0]], t_uniform)
print("\nSINDy model:")
model.print()
print(f"MSE = {mean_squared_error(L_avg_smooth, pred_max)}")
c0, c1, c2 = model.coefficients().flatten()
print(f"c0={c0:.3g}, c1={c1:.3g}, c2={c2:.3g}")

plt.figure()
plt.plot(t_uniform, L_max_avg,      'b',  label='⟨max L⟩ raw')
plt.plot(t_uniform, L_max_smooth,   'r',  label='⟨max L⟩ smoothed')
plt.plot(t_uniform, pred_max,       'k--',label='SINDy sim')
plt.legend()
plt.xlabel('t')
plt.ylabel('⟨max L⟩')

plt.figure()
plt.plot(t_uniform, dLmax_dt,                          'b',  label='finite diff')
plt.plot(t_uniform,
         model.predict(L_max_smooth.reshape(-1,1)),  'r--',label='SINDy')
plt.legend()
plt.xlabel('t')
plt.ylabel('d⟨max L⟩/dt')
plt.show()
