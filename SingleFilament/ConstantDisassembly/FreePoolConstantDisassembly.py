"""
simulate_continuous_filament.py

Simulate continuous‐time filament growth and severing via a stochastic model,
analyze average behavior across multiple trajectories,
identify governing equations with SINDy, and visualize results.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error


def simulate_filament_growth_continuous(
    N_tot: int,
    r: float,
    gamma: float,
    max_events: int,
    T_max: float = np.inf,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run a continuous‐time stochastic simulation of filament growth & severing.

    Parameters
    ----------
    N_tot : int
        Total number of monomers.
    r : float
        Growth rate coefficient.
    gamma : float
        Constant severing (decay) rate.
    max_events : int
        Maximum number of reaction events.
    T_max : float, optional
        Maximum simulation time (default: infinity).

    Returns
    -------
    T_list : np.ndarray
        Event times.
    L_list : np.ndarray
        Filament lengths at those times.
    """
    L = 1
    T = 0.0
    T_list = [T]
    L_list = [L]
    events = 0

    while events < max_events and T < T_max:
        available = max(N_tot - L, 0)
        k_grow = r * available
        k_decay = gamma
        k_total = k_grow + k_decay

        if k_total <= 0:
            break

        # Time to next event
        tau = np.random.exponential(1 / k_total)
        T += tau

        # Choose event
        if np.random.rand() < (k_decay / k_total):
            # Severing
            L = max(L - 1, 1)
        else:
            # Growth
            L += 1

        T_list.append(T)
        L_list.append(L)
        events += 1

    return np.array(T_list), np.array(L_list)


def run_multiple_trajectories(
    num_trajectories: int,
    N_tot: int,
    r: float,
    gamma: float,
    max_events: int,
    T_max: float = np.inf,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    """
    Simulate many independent filament trajectories.

    Parameters
    ----------
    num_trajectories : int
        How many trajectories to run.
    N_tot, r, gamma, max_events, T_max
        As in simulate_filament_growth_continuous.

    Returns
    -------
    trajectories : list of (times, lengths)
    final_lengths : np.ndarray
    final_times : np.ndarray
    """
    trajectories = []
    final_lengths = []
    final_times = []

    for _ in range(num_trajectories):
        T_arr, L_arr = simulate_filament_growth_continuous(
            N_tot, r, gamma, max_events, T_max
        )
        trajectories.append((T_arr, L_arr))
        final_lengths.append(L_arr[-1])
        final_times.append(T_arr[-1])

    return trajectories, np.array(final_lengths), np.array(final_times)


def preprocess_data(
    time: np.ndarray, data: np.ndarray, window: int = 51, polyorder: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """
    Smooth data and compute its derivative.

    Parameters
    ----------
    time : np.ndarray
    data : np.ndarray
    window : int
        Savitzky–Golay window length.
    polyorder : int
        Savitzky–Golay polynomial order.

    Returns
    -------
    smoothed : np.ndarray
    derivative : np.ndarray
    """
    smoothed = savgol_filter(data, window_length=window, polyorder=polyorder)
    dt = time[1] - time[0]
    derivative = np.gradient(smoothed, dt)
    return smoothed, derivative


def main():
    # Simulation parameters
    MaxS = 100_000
    N_tot = 1_000
    r = 0.3
    gamma = 225.0
    num_trajectories = 100

    # Run many trajectories
    trajectories, final_lengths, final_times = run_multiple_trajectories(
        num_trajectories, N_tot, r, gamma, MaxS
    )

    # Define a common time grid up to the shortest trajectory
    T_end = final_times.min()
    T_grid = np.linspace(0, T_end, 1_000)

    # Interpolate and average
    all_interp = np.vstack(
        [np.interp(T_grid, T_arr, L_arr) for T_arr, L_arr in trajectories]
    )
    avg_traj = all_interp.mean(axis=0)

    # Smooth & differentiate
    smoothed_traj, dL_dt = preprocess_data(T_grid, avg_traj)

    # Fit SINDy with a linear + bias library
    X = smoothed_traj.reshape(-1, 1)
    dX_dt = dL_dt.reshape(-1, 1)
    poly_lib = ps.PolynomialLibrary(degree=1, include_bias=True)
    model = ps.SINDy(optimizer=ps.STLSQ(threshold=1e-8), feature_library=poly_lib)
    model.fit(X, t=T_grid[1] - T_grid[0], x_dot=dX_dt)

    # Simulate & compute error
    X_pred = model.simulate(X[0], T_grid)
    mse = mean_squared_error(smoothed_traj, X_pred)

    # Extract parameters
    c0, c1 = model.coefficients().flatten()
    r_eff = -c1
    gamma_eff = r_eff * N_tot - c0
    L_ss = N_tot - gamma_eff / r_eff

    # Print results
    print(f"SINDy MSE = {mse:.6f}\n")
    print("Discovered model:")
    model.print()
    print("\nInterpreted parameters:")
    print(f"  c0 (bias)   = {c0:.6f}")
    print(f"  c1 (linear) = {c1:.6f}")
    print(f"  -> r_eff    = {r_eff:.6f}")
    print(f"  -> gamma_eff= {gamma_eff:.6f}")
    print(f"  Steady state L_ss = {L_ss:.6f}")

    # Plot individual trajectories
    plt.figure(figsize=(10, 6))
    for idx, (T_arr, L_arr) in enumerate(trajectories[:10]):
        plt.plot(T_arr, L_arr, alpha=0.6, label=f"Traj {idx+1}")
    plt.xlabel("Time")
    plt.ylabel("Filament Length")
    plt.title("Sample Individual Trajectories")
    plt.grid(True)
    plt.legend(fontsize="small")
    plt.tight_layout()
    plt.show()

    # Histogram of final lengths
    plt.figure(figsize=(8, 6))
    plt.hist(final_lengths, bins=30, density=True, alpha=0.7)
    plt.xlabel("Final Filament Length")
    plt.ylabel("Density")
    plt.title("Distribution of Final Filament Lengths")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Average vs. SINDy vs. theory
    theory = L_ss + (1 - L_ss) * np.exp(-r * T_grid)
    plt.figure(figsize=(10, 6))
    plt.plot(T_grid, avg_traj, label="Average Sim")
    plt.plot(T_grid, X_pred, "--", label="SINDy")
    plt.plot(T_grid, theory, "-.", label="Theory")
    plt.xlabel("Time")
    plt.ylabel("Filament Length")
    plt.title("Average Trajectory & Model Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Derivative comparison
    dL_dt_sindy = model.predict(X)
    plt.figure(figsize=(8, 5))
    plt.plot(T_grid, dL_dt, "-", label="Finite Difference")
    plt.plot(T_grid, dL_dt_sindy, "--", label="SINDy")
    plt.xlabel("Time (s)")
    plt.ylabel("d⟨L⟩/dt")
    plt.title("d⟨L⟩/dt: Finite Diff vs. SINDy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
