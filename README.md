# Two-Ion Spin-Boson Model Simulation

This project implements a simulation of a two-ion spin-boson model in a trapped-ion platform. It models the environment-mediated couplings between spins via the collective phonon modes of a linear ion chain.

## Features

- **Ion Chain Simulation**: Calculates equilibrium positions and transverse normal modes for a chain of $N=20$ $^{171}\mathrm{Yb}^+$ ions in an anharmonic axial potential.
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
1.  Initialize the ion chain ($N=20$) and calculate modes scaled to the 2.133-2.406 MHz band.
2.  Compute and plot coupling matrices $J_{ij}$ for "dense" and "sparse" spectral regimes.
    - Output: `J_ij_heatmaps.png`
3.  Simulate the time evolution of two central ions initialized in $|e,g\rangle$.
4.  Compute quantum correlations and plot them.
    - Output: `Dynamics.png`

## Physics Model

The system Hamiltonian is given by:

$$
H = \sum_{k} \omega_k a_k^{\dagger} a_k + \frac{\Delta}{2}\sum_{i} \sigma_{i,z} + \Omega \sum_{i,k} \eta_k b_{k,i} (\sigma^{+}_{i} a_k + \sigma^{-}_{i} a_k^{\dagger})
$$

We model the transverse modes of the linear chain, which naturally exhibit a high density of states near the Center-of-Mass (COM) mode at the top of the frequency band. This matches the "dense" regime description provided.

## File Structure

- `src/ion_chain.py`: Class `LinearChain` for ion positions and mode calculation.
- `src/physics.py`: Class `SpinBosonSystem` for Hamiltonian construction and dynamics.
- `src/main.py`: Main script to run the simulation and generate plots.
- `tests/`: Unit tests for the modules.
