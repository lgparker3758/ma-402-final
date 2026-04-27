from petsc4py import PETSc
import sys

class UserCtx:
    """
    User context for the SNES solver.

    Attributes
    ----------
    red1 : petsc4py.PETSc.DMDA
        DMDA for the design variable w.
    da1 : petsc4py.PETSc.DMDA
        DMDA for the state variable u.
    da2 : petsc4py.PETSc.DMDA
        DMDA for the adjoint variable lambda.
    packer : petsc4py.PETSc.DMComposite
        Composite manager to bundle the DMs together.
    """
    def __init__(self):
        self.red1 = None
        self.da1 = None
        self.da2 = None
        self.packer = None

def FormFunction(snes, U, F):
    """
    Evaluates the residual equations for the optimization problem.

    Parameters
    ----------
    snes : petsc4py.PETSc.SNES
        The nonlinear solver object.
    U : petsc4py.PETSc.Vec
        The global input vector containing [w, u, lambda].
    F : petsc4py.PETSc.Vec
        The global output vector where the residual is stored.

    Returns
    -------
    None
    """
    ctx = snes.getApplicationContext()
    packer = ctx.packer

    # 1. Create temporary local vectors
    V_local = packer.createLocalVec()
    F_local = packer.createLocalVec()

    # 2. Scatter global U into local V_local
    packer.globalToLocal(U, V_local)
    
    # Initialize local residual to zero
    F_local.set(0.0)

    # 3. Use the Context Manager! (The 'with' statement)
    with packer.getAccess(V_local) as v_locs, packer.getAccess(F_local) as f_locs:
        # Now we can unpack them safely
        vw, vu, vlam = v_locs
        vfw, vfu, vflam = f_locs

        # 4. Get arrays for math
        w_arr = vw.getArray()
        u_arr = ctx.da1.getVecArray(vu)
        lam_arr = ctx.da2.getVecArray(vlam)

        fw_arr = vfw.getArray()
        fu_arr = ctx.da1.getVecArray(vfu)
        fl_arr = ctx.da2.getVecArray(vflam)

        # Grid parameters
        corners, sizes = ctx.da1.getCorners()
        xs = corners[0] 
        xm = sizes[0]    
    
        N = ctx.da1.getSizes()[0]
        h = 1.0 / float(N - 1)
        d = 1.0 / h

        # --- MATH LOGIC ---

        # Equation for w (Design variable)
        if xs == 0:
            fw_arr[0] = -2.0 * d * lam_arr[0]

        # Equations for lambda (Adjoint variable)
        for i in range(xs, xs + xm):
            if i == 0:
                fl_arr[i] = h * u_arr[i] + 2.0 * d * lam_arr[i] - d * lam_arr[i+1]
            elif i == 1:
                fl_arr[i] = 2.0 * h * u_arr[i] + 2.0 * d * lam_arr[i] - d * lam_arr[i+1]
            elif i == N - 1:
                fl_arr[i] = h * u_arr[i] + 2.0 * d * lam_arr[i] - d * lam_arr[i-1]
            elif i == N - 2:
                fl_arr[i] = 2.0 * h * u_arr[i] + 2.0 * d * lam_arr[i] - d * lam_arr[i-1]
            else:
                fl_arr[i] = 2.0 * h * u_arr[i] - d * (lam_arr[i+1] - 2.0 * lam_arr[i] + lam_arr[i-1])

        # Equations for u (State variable)
        for i in range(xs, xs + xm):
            if i == 0:
                fu_arr[i] = 2.0 * d * (u_arr[i] - w_arr[0])
            elif i == N - 1:
                fu_arr[i] = 2.0 * d * u_arr[i]
            else:
                fu_arr[i] = -(d * (u_arr[i+1] - 2.0 * u_arr[i] + u_arr[i-1]) - 2.0 * h)

    # 6. Push local residual results into the global vector F
    packer.localToGlobal(F_local, F, PETSc.InsertMode.INSERT_VALUES)

    # 7. Destroy local vectors
    V_local.destroy()
    F_local.destroy()

def main():
    comm = PETSc.COMM_WORLD
    PETSc.Sys.Print("--- Starting PETSc Program ---", flush=True)

    ctx = UserCtx()

    # 1. Setup 'w'
    ctx.red1 = PETSc.DMDA().create(dim=1, sizes=[1], dof=1, stencil_width=0, comm=comm)
    ctx.red1.setUp()

    # 2. Setup 'u'
    ctx.da1 = PETSc.DMDA().create(dim=1, sizes=[5], dof=1, stencil_width=1, comm=comm)
    ctx.da1.setUp()

    # 3. Setup 'lambda'
    ctx.da2 = PETSc.DMDA().create(dim=1, sizes=[5], dof=1, stencil_width=1, comm=comm)
    ctx.da2.setUp()

    # 4. Setup Packer
    ctx.packer = PETSc.DMComposite().create(comm)
    ctx.packer.addDM(ctx.red1)
    ctx.packer.addDM(ctx.da1)
    ctx.packer.addDM(ctx.da2)

    # 5. Vectors
    U = ctx.packer.createGlobalVec()
    F = U.duplicate()
    U.set(0.0) 

    # 6. Solver
    snes = PETSc.SNES().create(comm)
    snes.setFunction(FormFunction, F)
    snes.setApplicationContext(ctx)
    snes.setFromOptions()

    snes.solve(None, U)
    
    # Results
    with ctx.packer.getAccess(U) as u_parts:
        w_final = u_parts[0].getArray()[0]
        u_final = ctx.da1.getVecArray(u_parts[1])
        
        PETSc.Sys.Print(f"\nFinal Optimized w: {w_final:.6f}", flush=True)
        PETSc.Sys.Print(f"State u at middle node: {u_final[2]:.6f}", flush=True)

    # Cleanup
    snes.destroy()
    U.destroy()
    F.destroy()
    ctx.red1.destroy()
    ctx.da1.destroy()
    ctx.da2.destroy()
    ctx.packer.destroy()

if __name__ == "__main__":
    main()