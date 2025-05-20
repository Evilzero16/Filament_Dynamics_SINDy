import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import pysindy as ps
from sklearn.metrics import mean_squared_error
np.math = math


def name_constant(input_features):
    """Name for the constant feature."""
    return "1"


def name_quadratic(input_features):
    """Name for the quadratic feature."""
    return f"{input_features[0]}^2"


def simulate_filament(T_end, max_steps, num=1, r=0.3, s=0.0075, Ntot=1000):
    """
    Simulate filament growth and severing until time T_end.

    Parameters
    ----------
    T_end : float
        Final simulation time (seconds).
    max_steps : int
        Maximum number of reaction events to simulate.
    num : int
        Number of filaments (unused, defaults to 1).
    r : float
        Monomer addition (growth) rate.
    s : float
        Severing rate per filament unit length.
    Ntot : int
        Total number of available monomers.

    Returns
    -------
    T : np.ndarray
        Times of each event up to T_end.
    m : np.ndarray
        Filament lengths at each recorded time.
    """
    m = np.full(max_steps, np.nan)
    T = np.zeros(max_steps)
    m[0] = 1       # initial filament length
    monomers = Ntot

    for i in range(max_steps - 1):
        severing_rate = s * m[i]
        growth_rate = r * Ntot
        total_rate = severing_rate + growth_rate

        # Time to next reaction
        u_time = np.random.rand()
        tau = -np.log(u_time) / total_rate
        T[i + 1] = T[i] + tau

        # Determine which reaction occurs
        u_event = np.random.rand()
        if u_event <= (severing_rate / total_rate):
            # Severing event
            if m[i] > 1:
                # Amount severed (integer)
                severed = np.floor((u_event / (severing_rate / total_rate)) * m[i])
                m[i + 1] = m[i] - severed
                monomers += severed
            else:
                m[i + 1] = m[i]
        else:
            # Growth event
            m[i + 1] = m[i] + 1
            monomers -= 1

        # Stop if we've reached the end time
        if T[i + 1] >= T_end:
            return T[: i + 2], m[: i + 2]

    return T, m


def main():
    # Simulation parameters
    T_end = 30
    max_steps = 300_000
    r = 0.3
    s = 0.0075
    Ntot = 1_000
    num_traj = 5_000
    N_points = 1_000

    # Uniform time grid for interpolation
    T_uniform = np.linspace(0, T_end, N_points)
    L_all = np.zeros((num_traj, N_points))

    # Run trajectories and interpolate
    for traj in range(num_traj):
        T_data, L_data = simulate_filament(T_end, max_steps, r=r, s=s, Ntot=Ntot)
        interp = interp1d(T_data, L_data, kind="linear", fill_value="extrapolate")
        L_all[traj, :] = interp(T_uniform)

    # Compute average and smooth
    L_avg = L_all.mean(axis=0)
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

    # Define a custom library: constant and quadratic
    library_funcs = [lambda x: np.ones_like(x), lambda x: x**2]
    function_names = [name_constant, name_quadratic]
    library = ps.CustomLibrary(library_functions=library_funcs,
                               function_names=function_names)

    optimizer = ps.STLSQ(threshold=1e-6, alpha=1e-3)
    model = ps.SINDy(feature_library=library, optimizer=optimizer)

    # Estimate derivatives via finite difference
    dL_dt_fd = np.gradient(L_avg_smooth, T_uniform)

    # Fit the model
    model.fit(X, t=dt, x_dot=dL_dt_fd.reshape(-1, 1))

    # Simulate and compute MSE
    pred = model.simulate(np.array([L_avg_smooth[0]]), T_uniform)
    mse = mean_squared_error(L_avg_smooth, pred)
    print(f"Mean Squared Error (MSE): {mse:.6f}\n")

    # Display discovered equation
    print("Discovered Governing Equation for ⟨L⟩ (Smoothed & Averaged Data):")
    model.print()

    # Extract and interpret coefficients
    coeffs = model.coefficients().flatten()
    names = model.get_feature_names()
    c0 = coeffs[names.index("f0(x0)")]
    c2 = coeffs[names.index("f1(x0)")]

    kprime_plus = c0 / Ntot
    L_ss_est = math.sqrt(-c0 / c2)
    L_ss_theory = math.sqrt((math.pi * r * Ntot) / (2 * s))
    f_est = (L_ss_est**2 * s) / (kprime_plus * Ntot)
    k_s = math.sqrt((kprime_plus * Ntot * s) / f_est)

    print(f"\nk'_+ (estimated) = {kprime_plus:.6f}")
    print(f"  -> Implied L_ss = {L_ss_est:.6f}")
    print(f"Theoretical L_ss = {L_ss_theory:.6f}")
    print(f"  -> Implied f = {f_est:.6f}")
    print(f"  -> Implied k_s = {k_s:.6f}")

    # Plot derivative comparison
    dL_dt_sindy = model.predict(X)
    plt.figure(figsize=(8, 5))
    plt.plot(T_uniform, dL_dt_fd, label="Finite Difference d⟨L⟩/dt")
    plt.plot(T_uniform, dL_dt_sindy, linestyle="--", label="SINDy d⟨L⟩/dt")
    plt.xlabel("Time (s)")
    plt.ylabel("d⟨L⟩/dt")
    plt.title("Derivative Comparison")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
