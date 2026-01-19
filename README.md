# Two-Ion Spin-Boson Model Simulation

This project implements a simulation of a two-ion spin-boson model in a trapped-ion platform. It models the environment-mediated couplings between spins via the collective phonon modes of a linear ion chain.

## Features

- **Ion Chain Simulation**: Calculates equilibrium positions and transverse normal modes for a linear $^{171}\mathrm{Yb}^+$ ion chain.
- **Spin-Boson Physics**:
  - Calculates the mode-resolved spin-phonon coupling matrix $J_{ij}(\omega)$.
  - Constructs the Hamiltonian for a chosen pair of spins coupled to the phonon bath.
  - Simulates the exact quantum dynamics in the single-excitation subspace.
- **Analysis**:
  - Computes Connected Correlator $C_{12}(t)$.
  - Computes Entanglement of Formation (EoF).
  - Computes Quantum Discord (QD).
- **Visualization**: Generates heatmaps of coupling strengths and time-evolution plots of correlations.

## Important Note

I am an AI assistant and **cannot connect directly to your Overleaf account** or access external user accounts. I have provided the complete Python source code here for you to run locally or integrate into your workflow.

## Installation

1.  Clone this repository.
2.  Install the required dependencies:
    ```bash
    pip install numpy scipy matplotlib
    ```

## Usage

Run the main simulation script:

```bash
python src/main.py
```

This will perform the following steps:
1.  Initialize the ion chain using provided experimental data ($N=10$, specific distances and frequencies).
2.  Compute and plot the coupling matrix $J_{ij}$ at the center of the phonon band.
    - Output: `J_ij_10ions.png`
3.  Simulate the time evolution of the two central ions (indices 4 and 5) initialized in $|e,g\rangle$.
4.  Compute quantum correlations and plot them.
    - Output: `Dynamics_10ions.png`

## Physics Model

The system Hamiltonian is given by:

$$
H = \sum_{k} \omega_k a_k^{\dagger} a_k + \frac{\Delta}{2}\sum_{i} \sigma_{i,z} + \Omega \sum_{i,k} \eta_k b_{k,i} (\sigma^{+}_{i} a_k + \sigma^{-}_{i} a_k^{\dagger})
$$

The simulation uses transverse phonon modes derived from the provided ion distances and frequencies.

## File Structure

- `src/ion_chain.py`: Class `LinearChain` for ion positions and mode calculation.
- `src/physics.py`: Class `SpinBosonSystem` for Hamiltonian construction and dynamics.
- `src/main.py`: Main script to run the simulation and generate plots.
- `tests/`: Unit tests for the modules.
