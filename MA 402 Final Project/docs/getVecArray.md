# Function Documentation: `PETSc.DMDA.getVecArray`

### 1. Description
The `getVecArray` method provides a multi-dimensional, NumPy-compatible view of a PETSc Vector’s data. This is a critical utility for structured grids (DMDA), as it maps the flat memory layout of a PETSc `Vec` into a structured array that matches the grid's dimensions (1D, 2D, or 3D). This allows for intuitive indexing using grid coordinates rather than manual memory offsets.

### 2. Parameters and Return Types
* **vec** (*PETSc.Vec*): The vector to be accessed. 
    * *Note:* To access "ghost points" (neighboring data from other processors), this should be a **local vector** obtained via `dm.getLocalVec()`.
* **readonly** (*bool, optional*): If set to `True`, the returned NumPy array will be read-only to prevent accidental modification of the underlying PETSc vector. Default is `False`.
* **Returns**: (*numpy.ndarray*) A multi-dimensional array view of the vector data. Changes made to this array (if not `readonly`) are reflected directly in the PETSc vector.

### 3. Mathematical Context
In scientific computing, a physical domain $\Omega$ is discretized into a grid. While a PETSc `Vec` stores these values in a 1D contiguous array for linear algebra efficiency, physical operators (like the Laplacian) are defined by spatial relationships. 

For a 2D grid, `getVecArray` performs the transformation from a flat index $k$ to spatial indices $(i, j)$:
$$u_{i,j} = \text{Vec}[k]$$

This allows for the direct implementation of finite difference stencils. For example, the 5-point Laplacian $\nabla^2 u \approx \Delta u$ at grid point $(i,j)$ is implemented as:
$$\Delta u \approx \frac{u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j}}{h^2}$$

### 4. Source Code Archaeology
* **C Header:** [`include/petscdmda.h`](https://gitlab.com/petsc/petsc/-/blob/main/include/petscdmda.h?ref_type=heads#L127)
* **C Source:** [`src/dm/impls/da/dagetarray.c`](https://gitlab.com/petsc/petsc/-/blob/main/src/dm/impls/da/dagetarray.c?ref_type=heads#L43)
* **The Cython Bridge:** This method is defined in `src/binding/petsc4py/src/petsc4py/PETSc/DMDA.pyx`. It acts as a high-level wrapper for the C routine `DMDAVecGetArray`. The bridge logic (specifically in `lpldmda.pxi`) handles the complexity of detecting the DMDA dimension and automatically selecting the correct C call (e.g., `DMDAVecGetArray2d`) before returning a NumPy-managed memory view to the user.

### 5. Minimal Working Example (MWE)
```python
# Context: Evaluating a residual on a 2D structured grid
def formFunction(snes, X, F, ctx):
    # 1. Scatter global solution to local vector (to get ghost points)
    localX = ctx.da.getLocalVec()
    ctx.da.globalToLocal(X, localX)

    # 2. Get the array view using the bridge function
    x_arr = ctx.da.getVecArray(localX)
    f_arr = ctx.da.getVecArray(F)

    # 3. Access data using grid indices (i, j)
    # This matches the mathematical formulation u_{i,j}
    for j in range(ys, ye):
        for i in range(xs, xe):
            f_arr[i, j] = x_arr[i, j] - 1.0 # Simple example operation
            
    # 4. Clean up
    ctx.da.restoreLocalVec(localX)