# Filament_Dynamics_SINDy
We investigate three distinct length control mechanisms for a single filament:
(1) constant disassembly under a limited monomer pool, (2) severing in an
unlimited (free) pool, and (3) severing under a limited pool. For each
mechanism, we perform stochastic simulations using the Gillespie algorithm
across numerous trajectories. We then compute the ensemble-averaged
filament length, apply a Savitzky–Golay filter to smooth the dynamics, and
use Sparse Identification of Nonlinear Dynamics (SINDy) for model
inference. By fitting models that include only a constant term and the
lowest-order length-dependent term predicted by theory, SINDy accurately
recovers the theoretical coefficients within approximately 10%. These
data-driven results validate the underlying rate equations and suggest that
high-throughput single-filament experiments could reliably distinguish
among the three control mechanisms.
