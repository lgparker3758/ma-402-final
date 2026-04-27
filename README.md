# MA-402-Final
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
* [`MA402_FinalProject.py`](https://github.com/lgparker3758/MA-402-Final-Project/blob/main/MA402_FinalProject.py): The core PETSc solver and optimization logic.
* [`MA402_visualizations.ipynb`](https://github.com/lgparker3758/MA-402-Final-Project/blob/main/MA402_visualizations.ipynb): Jupyter Notebook for analyzing solver convergence and results.
* `docs/`: Deep-dive technical documentation for PETSc methods.
    * [`getVecArray.ipynb`](https://github.com/lgparker3758/MA-402-Final-Project/blob/main/getVecArray.md): Handling local vector data.
    * [`globalToLocal.ipynb`](https://github.com/lgparker3758/MA-402-Final-Project/blob/main/globalToLocal.md): Managing ghost point synchronization.
    * [`getAccess.ipynb`](https://github.com/lgparker3758/MA-402-Final-Project/blob/main/getAccess.md): Using DMComposite context managers.
 
## AI Translation Experience
I used Gemini to help translate C code into Python code. It was an "ok" experience for someone who has never worked with C code and isn't extremely confident with Python. I felt that Gemini was better at writing code than ChatGPT, but it still created a lot of errors. The biggest problem I had was with trying to get VS Code and Python to work correctly. I consistently had issues with path directory, being told pip wasn't installed, being told ipy kernel wasn't installed, and more. 

## Installation & Usage
Ensure you have `petsc4py` installed in your WSL/Linux environment.

### Run the Solver
To run the solver in serial:
```bash
python3 MA402_Final.py
