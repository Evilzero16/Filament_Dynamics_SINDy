import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pathlib

num         = 6          # number of filaments in pool
r           = 0.01       # growth rate
s           = 0.074      # severing rate coefficient
Ntot        = 100_000    # total monomers in pool
MaxS        = 3_000_000  # maximum simulation steps
MaxTraj     = 500        # Monte-Carlo trajectories
t1_values   = np.arange(0, 20.0 + 1e-9, 0.1)  # evaluation times (inclusive)

# Pre-allocate result containers (len(t1_values) == 201)
Max_mean_values     = np.zeros_like(t1_values)
Max_variance_values = np.zeros_like(t1_values)
Min_mean_values     = np.zeros_like(t1_values)
Min_variance_values = np.zeros_like(t1_values)
Single_mean_values  = np.zeros_like(t1_values)
Single_variance_values = np.zeros_like(t1_values)
Single_SD_values    = np.zeros_like(t1_values)
Tot_mean_values     = np.zeros_like(t1_values)
Tot_variance_values = np.zeros_like(t1_values)
Tot_SD_values       = np.zeros_like(t1_values)

# Convenience: index offset because Python is 0-based but lengths start at 1
OFFSET = -1

for t_idx, t1 in enumerate(t1_values):

    # Probability histograms (index = length-1)
    p_max = np.zeros(Ntot, dtype=np.int64)
    p_min = np.zeros(Ntot, dtype=np.int64)
    p_single = np.zeros(Ntot, dtype=np.int64)
    p_tot   = np.zeros(Ntot, dtype=np.int64)

    for traj in range(MaxTraj):
        # --- trajectory initialisation --- #
        m      = np.full((num, MaxS+1), np.nan, dtype=np.int32)  # filament lengths
        mtot   = np.full(MaxS+1, np.nan, dtype=np.int32)         # total length
        mmax   = np.full(MaxS+1, np.nan, dtype=np.int32)         # max length
        mmin   = np.full(MaxS+1, np.nan, dtype=np.int32)         # min length
        T      = np.zeros(MaxS+1, dtype=np.float64)              # absolute time
        tau    = np.zeros(MaxS,   dtype=np.float64)              # step durations

        m[:, 0]   = 1
        mtot[0]   = num
        mmax[0]   = 1
        mmin[0]   = 1
        monomers  = Ntot                                        # free pool

        # --- kinetic Monte-Carlo loop --- #
        for step in range(MaxS):
            fil = np.random.randint(0, num)                # which filament reacts?

            # Propensities
            k1 = s * m[fil, step]                          # severing rate
            k2 = r * Ntot                                  # growth rate (unlimited pool)
            k0 = k1 + k2
            if k0 == 0:
                break

            # Time increment (Gillespie)
            tau[step] = np.random.exponential(1.0 / k0)
            T[step+1] = T[step] + tau[step]

            # Decide reaction channel
            if np.random.rand() <= k1 / k0:                # severing
                if m[fil, step] == 1:
                    m[fil, step+1] = m[fil, step]          # no sever if length==1
                else:
                    # sever point chosen uniformly along filament
                    sever_point = int(np.floor(np.random.rand() * m[fil, step]))
                    m[fil, step+1] = m[fil, step] - sever_point
                    monomers += sever_point
            else:                                          # growth
                m[fil, step+1] = m[fil, step] + 1
                monomers -= 1

            # All other filaments keep previous length
            m[:, step+1] = np.where(np.arange(num)[:, None] == fil, m[:, step+1], m[:, step])

            mtot[step+1] = m[:, step+1].sum()
            mmax[step+1] = m[:, step+1].max()
            mmin[step+1] = m[:, step+1].min()

            if T[step+1] >= t1:
                break

        # Use the final state at index (step+1)
        length_single = int(m[0, step+1])
        length_max    = int(mmax[step+1])
        length_min    = int(mmin[step+1])
        length_tot    = int(mtot[step+1])

        # Histogram increment (shift by OFFSET = -1)
        p_single[length_single + OFFSET] += 1
        p_max[length_max + OFFSET]       += 1
        p_min[length_min + OFFSET]       += 1
        p_tot[length_tot + OFFSET]       += 1

    x = np.arange(1, Ntot+1)                        # possible lengths

    for arr, mean_store, var_store, sd_store in [
        (p_max,    Max_mean_values,  Max_variance_values, None),
        (p_min,    Min_mean_values,  Min_variance_values, None),
        (p_single, Single_mean_values, Single_variance_values, Single_SD_values),
        (p_tot,    Tot_mean_values,   Tot_variance_values,  Tot_SD_values)]:

        if arr.sum() == 0:
            continue
        p = arr / arr.sum()
        mean = np.dot(x, p)
        var  = np.dot(x**2, p) - mean**2
        sd   = np.sqrt(var)

        mean_store[t_idx] = mean
        var_store[t_idx]  = var
        if sd_store is not None:
            sd_store[t_idx] = sd

def exp_model(t, Lk, k):
    return Lk * (1.0 - np.exp(-k * t))

# — Individual filament fit — #
params_single, _ = curve_fit(exp_model, t1_values, Single_mean_values - 1,
                             p0=[Single_mean_values[-1], 0.1])
Lk_single, k_single = params_single
print(f"Fitted LAvg: {Lk_single:.6f}")
print(f"Fitted kAvg: {k_single:.6f}")

plt.figure(figsize=(7,5))
plt.plot(t1_values, Single_mean_values-1, '.', ms=6, lw=0, label='Individual Filament Sev Sim')
plt.plot(t1_values, exp_model(t1_values, *params_single), '-', lw=2, label='Fitted Curve')
plt.xlabel('time')
plt.ylabel('Length (monomers)')
plt.legend()
plt.tight_layout()

# — Bundle (max) fit — #
params_max, _ = curve_fit(exp_model, t1_values, Max_mean_values - 1,
                          p0=[Max_mean_values[-1], 0.1])
Lk_max, k_max = params_max
print(f"Fitted Lk1Max: {Lk_max:.6f}")
print(f"Fitted k1Max:  {k_max:.6f}")

plt.figure(figsize=(7,5))
plt.plot(t1_values, Max_mean_values-1, '.', ms=6, lw=0, label='Bundle Sev Sim')
plt.plot(t1_values, exp_model(t1_values, *params_max), '-', lw=2, label='Fitted Curve')
plt.xlabel('time')
plt.ylabel('Bundle Length')
plt.legend()
plt.tight_layout()

plt.show()

out_dir = pathlib.Path(r"E:\RIT\Researcher\Hrishit\Bundle\Sev")
out_dir.mkdir(parents=True, exist_ok=True)

np.savetxt(out_dir / "Fil6Time.dat",          t1_values, fmt="%.6f")
np.savetxt(out_dir / "Fil6SingleLength.dat",  Single_mean_values, fmt="%.6f")
np.savetxt(out_dir / "Fil6BundleLength.dat",  Max_mean_values, fmt="%.6f")
