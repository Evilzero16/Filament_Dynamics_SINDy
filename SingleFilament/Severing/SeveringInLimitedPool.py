"""
simulate_filament.py

Simulates filament growth and severing using a stochastic model,
plots averaged filament length, smooths the data, and discovers
governing equations via SINDy.
"""

import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import pysindy as ps
from sklearn.metrics import mean_squared_error

np.math = math


def simulate_filament(T_end, MaxS, num=1, r=0.3, s=0.0075, Ntot=1000):
    """
    Simulate filament growth and severing until time T_end.

    Parameters:
        T_end (float): Final simulation time (seconds).
        MaxS (int): Maximum simulation steps.
        num (int): Number of filaments (default 1).
        r (float): Growth rate.
        s (float): Severing rate.
        Ntot (int): Total number of available monomers.

    Returns:
        tuple:
            T (np.ndarray): Time array up to T_end.
            m (np.ndarray): Filament length array for the simulated filament.
    """
    m = np.full(MaxS, np.nan)
    T = np.zeros(MaxS)
    m[0] = 1
    monomers = Ntot

    for i in range(MaxS - 1):
        s1 = s * m[i]
        k2 = r * (Ntot - m[i])
        k1 = s1
        k0 = k1 + k2

        tau = (1.0 / k0) * np.log(1.0 / np.random.rand())
        T[i + 1] = T[i] + tau

        if np.random.rand() <= (k1 / k0):
            if m[i] == 1:
                m[i + 1] = m[i]
            else:
                s1_new = np.floor(np.random.rand() / (k1 / k0) * m[i])
                m[i + 1] = m[i] - s1_new
                monomers += s1_new
        else:
            m[i + 1] = m[i] + 1
            monomers -= 1

        if T[i + 1] >= T_end:
            return T[: i + 2], m[: i + 2]

    return T, m


def main():
    """Run simulation, apply SINDy, and plot results."""
    # Simulation parameters
    T_end = 30
    MaxS = 300_000
    num_traj = 5_000
    r = 0.3
    s = 0.0075
    Ntot = 1_000
    N_points = 1_000

    T_uniform = np.linspace(0, T_end, N_points)
    L_all = np.zeros((num_traj, N_points))

    for traj in range(num_traj):
        T_data, L_data = simulate_filament(T_end, MaxS, num=1, r=r, s=s, Ntot=Ntot)
        interp_func = interp1d(T_data, L_data, kind="linear", fill_value="extrapolate")
        L_all[traj, :] = interp_func(T_uniform)

    # Average and smooth the data
    L_avg = np.mean(L_all, axis=0)
    L_avg_smooth = savgol_filter(L_avg, window_length=51, polyorder=3)

    # Plot averaged vs. smoothed data
    plt.figure(figsize=(8, 5))
    plt.plot(T_uniform, L_avg, label="Averaged Data")
    plt.plot(T_uniform, L_avg_smooth, label="Smoothed Data")
    plt.xlabel("Time (s)")
    plt.ylabel("Average Filament Length ⟨L⟩")
    plt.title("Averaged vs. Smoothed Data")
    plt.legend()
    plt.show()

    # Prepare data for SINDy
    X = L_avg_smooth.reshape(-1, 1)
    dt = T_uniform[1] - T_uniform[0]
    library = ps.PolynomialLibrary(degree=2, include_bias=True)
    optimizer = ps.STLSQ(threshold=1e-8, alpha=1e-3)
    model = ps.SINDy(feature_library=library, optimizer=optimizer)

    model.fit(X, t=dt, x_dot=np.gradient(L_avg_smooth, T_uniform))
    pred = model.simulate(np.array([L_avg_smooth[0]]), T_uniform)
    mse = mean_squared_error(L_avg_smooth, pred)
    print(f"Mean Squared Error (MSE): {mse:.6f}\n")

    print("Discovered Governing Equation for ⟨L⟩:")
    model.print()

    # Interpret coefficients
    coeffs = model.coefficients().flatten()
    feature_names = model.get_feature_names()
    c0 = coeffs[feature_names.index("1")]
    c1 = coeffs[feature_names.index("x0")]
    c2 = coeffs[feature_names.index("x0^2")]

    kprime_plus = c0 / Ntot
    L_ss_est = (-c1 - math.sqrt(c1**2 - 4 * c0 * c2)) / (2 * c2)
    L_ss_theory = (
        (-math.pi * r) + math.sqrt((math.pi**2) * r**2 + 8 * math.pi * s * r * Ntot)
    ) / (4 * s)
    k_lp = math.sqrt((kprime_plus * Ntot * s) * 2 / math.pi)

    print(f"k'_+ (estimated) = {kprime_plus:.6f}")
    print(f"  -> Implied L_ss = {L_ss_est:.6f}")
    print(f"Theoretical L_ss = {L_ss_theory:.6f}")
    print(f"  -> Implied k_lp = {k_lp:.6f}")

    # Plot derivative comparison
    plt.figure(figsize=(8, 5))
    plt.plot(
        T_uniform,
        np.gradient(L_avg_smooth, T_uniform),
        label="Finite Difference d⟨L⟩/dt",
    )
    plt.plot(
        T_uniform,
        model.predict(X),
        linestyle="--",
        label="SINDy d⟨L⟩/dt",
    )
    plt.xlabel("Time (s)")
    plt.ylabel("d⟨L⟩/dt")
    plt.title("Derivative Comparison")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
