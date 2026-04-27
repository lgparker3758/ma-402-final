# Parallel PDE-Constrained Optimization Solver

This project implements a parallel nonlinear solver for a 1D PDE-constrained optimization problem using **PETSc** and **petsc4py**. The solver uses a full-space (one-shot) approach to simultaneously solve for the design variable ($w$), state variable ($u$), and adjoint variable ($\lambda$).

## Overview
The solver targets a distributed mesh to find the optimal design that minimizes a cost functional subject to a steady-state Poisson equation.

### Key Features
* **Parallel Execution:** Scalable numerical operations via MPI.
* **DMComposite:** Efficient management of multiple physics fields ($w, u, \lambda$).
* **Nonlinear Solver:** Utilizes PETSc's `SNES` (Scalable Nonlinear Equations Solvers) interface.
* **Optimized Memory:** Uses context managers for automatic memory restoration of PETSc vectors.

## Project Structure
* `main_script.py`: The core PETSc solver and optimization logic.
* `visualization.ipynb`: Jupyter Notebook for analyzing solver convergence and results.
* `docs/`: Deep-dive technical documentation for PETSc methods.
    * `getVecArray.md`: Handling local vector data.
    * `globalToLocal.md`: Managing ghost point synchronization.
    * `getAccess.md`: Using DMComposite context managers.

## Installation & Usage
Ensure you have `petsc4py` installed in your WSL/Linux environment.

### Run the Solver
To run the solver in serial:
```bash
python3 MA402_Final.py
