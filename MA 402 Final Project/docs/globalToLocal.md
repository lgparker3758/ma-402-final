# Function Documentation: `PETSc.DM.globalToLocal`

### 1. Description
The `globalToLocal` method communicates data from a **global vector** to a **local vector**. This communication is mandatory in parallel computing before any updation of array elements can occur.

### 2. Parameters and Return Types
* **gvec** (*PETSc.Vec*): The input global vector containing the current solution state across all processors.
* **lvec** (*PETSc.Vec*): The output local vector that will be populated with the global data plus the necessary ghost point values from neighbors.
* **addv** (*PETSc.InsertMode*, optional): Specifies how to handle the data (e.g., `PETSc.InsertMode.INSERT_VALUES`). Defaults to `INSERT_VALUES`.
* **Returns**: `None`. The operation modifies the `lvec` in place.

### 3. Mathematical Context
In parallel domain decomposition, each processor $k$ owns a subdomain $\Omega_k$. To compute a spatial derivative at the boundary of $\Omega_k$, the processor requires values from the interior of the neighboring subdomain $\Omega_{k+1}$. These overlapping values are known as **Ghost Points**.

Mathematically, if $V_g$ is the global solution vector, `globalToLocal` executes the scatter operation:
$$V_{local, k} = \mathcal{S}_k V_g$$
where $\mathcal{S}_k$ is the operator that maps global degrees of freedom to the local memory space of processor $k$, including the required overlap. Without this call, boundary stencils would use zeroed or outdated data, resulting in incorrect physical residuals.

### 4. Source Code Archaeology
* **C Header:** [`include/petscdm.h#L134`](https://gitlab.com/petsc/petsc/-/blob/main/include/petscdm.h?ref_type=heads#L134)
* **C Source:** [`src/dm/interface/dm.c#L2892`](https://gitlab.com/petsc/petsc/-/blob/main/src/dm/interface/dm.c#L2892)
* **The Cython Bridge:** [`src/binding/petsc4py/src/petsc4py/PETSc/DM.pyx#L971`](https://gitlab.com/petsc/petsc/-/blob/main/src/binding/petsc4py/src/petsc4py/PETSc/DM.pyx#L971)

**Technical Insight:** The Cython bridge uses `CHKERR` to wrap the C function `DMGlobalToLocal`. This ensures that if the underlying MPI communication fails, a Python exception is raised. The bridge also resolve the Python `Vec` objects to pass the raw C pointers, which help interface with C style libraries, into the high-performance PETSc library.

### 5. Minimal Working Example (MWE)
```python
from petsc4py import PETSc

def main():
    comm = PETSc.COMM_WORLD
    # 1. Create a 1D DA
    da = PETSc.DMDA().create(dim=1, sizes=[10], comm=comm, stencil_width=1)
    da.setUp()

    g_sol = da.createGlobalVec()
    l_sol = da.getLocalVec()
    g_sol.set(1.0) 

    # 2. Sync
    da.globalToLocal(g_sol, l_sol)
    u_arr = da.getVecArray(l_sol)

    # 3. Get the range (Note the [0] at the end!)
    (xs, xe) = da.getRanges()[0]

    print(f"Processor {comm.getRank()} is handling indices from {xs} to {xe}")
    
    # Clean up
    da.restoreLocalVec(l_sol)
    g_sol.destroy()
    l_sol.destroy()
    da.destroy()

if __name__ == "__main__":
    main()
