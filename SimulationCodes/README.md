Planck-Scale Scalar Field Theory Simulations
This repository contains simulation codes for three papers:

Paper 1: Planck-Scale Scalar Field Theory for Vacuum Energy and Particle Phenomena
Paper 2: Planck-Scale Scalar Field Theory for Particle and Baryogenesis Phenomena
Paper 3: Planck-Scale Scalar Field Theory for Cosmology

Structure

SimulationCodes/paperX/: Code for each paper (X = 1, 2, 3).
SimulationCodes/common/: Shared utilities.
SimulationCodes/docs/: Setup, sensitivity analysis, and execution instructions.
SimulationCodes/requirements.txt: Dependencies.

Quick Start

Install dependencies:pip install -r SimulationCodes/requirements.txt


Compile Cython:cythonize -i SimulationCodes/paper1/lattice_cy.pyx
cythonize -i SimulationCodes/paper2/lattice_cy.pyx
cythonize -i SimulationCodes/paper3/lattice_cy.pyx


Run simulations (example for Paper 3):python SimulationCodes/paper3/vacuum_energy.py



Documentation

docs/simulation_setup.pdf: Environment setup.
docs/sensitivity_analysis.pdf: Parameter sensitivity.
docs/execution_instructions.md: Execution commands.

Data
Outputs saved in SimulationCodes/paperX/data/.
License
All Rights Reserved by Masayuki Note
