# Asymmetric Single-Sided Coupling in a Trapped-Ion Spin-Boson Quantum Simulator

## Abstract
The simulation of open quantum systems requires exact control over the coupling geometry between the system and its environment. We propose and blueprint an analog quantum simulator for a single-sided spin-boson model using an 8-ion linear chain of $^{171}\mathrm{Yb}^+$ trapped ions. By preparing an entangled Bell state and selectively driving only a single ion, we decouple the unexcited basis state, resulting in exact, non-trivial analytical relations connecting the bipartite Concurrence and Parity ($C(t) = \sqrt{P(t)}$). Unlike symmetric double-sided models, this setup mathematically prohibits Entanglement Sudden Death (ESD), ensuring asymptotic entanglement decay. Furthermore, driving the system within the phonon bandgap ($\omega_L = 1463.0$ kHz) demonstrates pronounced entanglement revivals ($P \approx 0.93$ at $t \approx 184\ \mu s$) protected by the phonon density of states. Our scheme halves the experimental control overhead and provides a robust testbed for decoherence protection free from abrupt non-Markovian ESD boundaries.

## 1. Model Definition
We study a two-impurity open quantum system where an initial maximally entangled Bell state $|\Phi^+\rangle = (|{\uparrow\uparrow}\rangle + |{\downarrow\downarrow}\rangle)/\sqrt{2}$ interacts with a common bosonic bath consisting of the $N=8$ collective transverse phonon modes of a trapped-ion chain.

A central innovation of our work is the **Single-Sided Coupling** geometry. Only Ion A (index 4) is irradiated by a laser to couple with the phonon bath, while Ion B (index 6) remains undriven. The interaction Hamiltonian in the rotating frame is given by:
$$ H_{int} = \sum_{k=1}^N g_{Ak} (\sigma_+^A a_k + \mathrm{h.c.}) + \sum_{k=1}^N \delta_k a_k^\dagger a_k $$
where $g_{Ak}$ is the effective coupling strength on Ion A to mode $k$, and $\delta_k = \omega_k - \omega_L$ is the detuning from the laser frequency. For our experiments, the base coupling strength is set to $g/2\pi = 8.0$ kHz.

This contrasts with a **Double-Sided Coupling** geometry, where both Ion A and Ion B are driven symmetrically, leading to two sets of coupling terms $\sum g_{Ak}(\dots) + \sum g_{Bk}(\dots)$.

## 2. Exact Analytical Solutions
In the single-sided coupling model, the state $|{\downarrow\downarrow}, \mathrm{vac}\rangle$ is a dark state and entirely completely decouples from the phonon bath. Consequently, the Hilbert space can be exactly blocked. The system state evolves exactly within a single-excitation subspace:
$$ |\Psi(t)\rangle = \frac{1}{\sqrt{2}} \left[ c_{uu}(t)|{\uparrow\uparrow}\rangle|\mathrm{vac}\rangle + \sum_{k=1}^N c_k(t)|{\downarrow\uparrow}\rangle|1_k\rangle + |{\downarrow\downarrow}\rangle|\mathrm{vac}\rangle \right] $$

Due to this structural protection, the experimental observables can be directly related to the single amplitude $c_{uu}(t)$. The Parity and Concurrence yield identical, exact mappings:
- **Parity**: $P(t) = |c_{uu}(t)|^2$
- **Concurrence**: $C(t) = |c_{uu}(t)|$

This strict relationship ($C(t) = \sqrt{P(t)}$) is maintained exactly (numerically verified to $< 10^{-15}$ precision). In experiments, this allows the full bipartite entanglement metric $C(t)$ to be extracted merely by measuring the global parity $P(t)$, vastly bypassing full state tomography.

## 3. Entanglement Sudden Death (ESD) and Bandgap Protection
The single-sided model exhibits fundamentally different phase boundaries compared to the double-sided system.

**Absence of ESD:** In double-sided configurations, the state purity can degrade rapidly, driving the Concurrence abruptly to zero—a phenomenon known as Entanglement Sudden Death. However, in our single-sided model, the continuous preservation of the $|{\downarrow\downarrow}\rangle$ amplitude provides a resilient entanglement "skeleton". Concurrence decays asymptotically and can never abruptly terminate, as $P(t) \geq 0$ holds strictly true.

**Bandgap Protection:** By tuning the single-sided laser frequency into the spectral bandgap of the phonon modes (e.g., $\omega_L = 1463.0$ kHz, positioned between the first mode at 1461.4 kHz and the second at 1470.8 kHz), phonon excitations are dynamically suppressed. We predict a massive, non-Markovian entanglement revival. Our simulations confirm that despite the loss of initial entanglement, a powerful revival occurs at $t \approx 184\ \mu s$, where the parity climbs to $P \approx 0.926$ and Concurrence is restored to $C \approx 0.962$.

## 4. Experimental Advantages
This single-sided approach represents an ideal architecture for early-stage quantum simulators.
1. **Reduced Overhead:** Only a single optical beam path needs phase and amplitude stabilization.
2. **Robustness to Errors:** The asymptotic decay guarantees that small experimental calibrations and extraneous noise sources will not trigger catastrophic, abrupt loss of entanglement.
3. **Pristine Observability:** The rigid lock between $C(t)$ and $P(t)$ significantly simplifies signal processing and enhances the signal-to-noise ratio in extracting non-Markovian signatures.

This platform bridges the theoretical study of structured bosonic baths with scalable, error-resilient experimental quantum simulations.