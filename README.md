# Quantum-Stochastic Simulator with Radon-Nikodym Noise Reweighting

A research-grade hybrid classical-quantum simulator integrating shot-based gradients, SDE-driven noise and RN reweighting

# Overview

This repository containss a modular, NumPy-first quantum machine learning simulator designed for research. It integrates: 

  - Variational quantum circuits
  - Shot-based gradient optimization
  - Stochastic noise modeled via SDEs
  - Radon-Nikodym (Girsanov) reweighting
  - Hybrid classical-quantum optimization
  - A clean, publishable architecture

The simulator is compact, mathematically rigorous and engineered for reproducibility. 

# Features

  - Variational circuits with statevector simulation
  - Shot-based gradients using parameter-shift
  - SDE driven noise with Euler-Maruyama sampling
  - RN reweighting via Girsanov's theorem
  - Hybrid optimization with ADAM
  - Simulator wrapper for clean experiment orchestration
  - API notebooks documenting each module
  - Single-file final product for easy execution
  - Modular multi-file version for research clarity

# Repository structure

quantum_stochastic_simulator/
    simulator_final.py
    circuit.ipynb
    noise.ipynb
    gradients.ipynb
    trainer.ipynb
    simulator.ipynb
    api/
        circuit_api.ipynb
        noise_api.ipynb
        gradients_api.ipynb
        trainer_api.ipynb
        simulator_api.ipynb simnulator
    experiments/
        vqe_rn_noise.ipynb
        hybrid_classifier.ipynb
    README.md
    requirements.txt

# Running the simulator

python simulator_final.py 

This runs a small hybrid regression experiment with RN-wweighted noisy gradients

# Theory

The simulator models noise as an SDE: 

dλ_t = a(λ_t)dt + b(λ_t)dWt

and reweights expectations under a different noise law Q: dQ/dP. 

This enables importance sampling of noisy quantum circuits without resimulation. 

# Example experiments

- Noisy VQE with RN reweighting
- Hybrid regression / classification
- Gradient variance analysis under shot noise
- Noise-law comparison  via RN weights

# License

MIT license.
