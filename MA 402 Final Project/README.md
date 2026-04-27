# Parallel PDE-Constrained Optimization Solver

This project finds a way to solve a complex, nonlinear differential 1D PDE-constrained optimization problem using **PETSc** and **petsc4py**. The solver uses an "all-at-once" approach by simultaneously solving for the design variable (inputs) ($w$), state variable (outputs) ($u$), and adjoint variable (sensitivities) ($\lambda$). 

## Overview
This project uses multiple processors to to optimize a physical setup. It balances getting to a specific goal while still following all rules of physics across a coordinate system.

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
 
## AI Translation Experience
I used Gemini to help translate C code into Python code. It was an "ok" experience for someone who has never worked with C code and isn't extremely confident with Python. I felt that Gemini was better at writing code than ChatGPT, but it still created a lot of errors. The biggest problem I had was with trying to get VS Code and Python to work correctly. I consistently had issues with path directory, being told pip wasn't installed, being told ipy kernel wasn't installed, and more. 

## Installation & Usage
Ensure you have `petsc4py` installed in your WSL/Linux environment.

### Run the Solver
To run the solver in serial:
```bash
python3 MA402_Final.py
