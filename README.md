"""python
import sys
import ctypes
from ctypes import POINTER, c_int, c_double, c_void_p, byref, CDLL
from mpi4py import MPI
import numpy as np

# 1. Load the hypre shared library
# Please change the path according to your environment
try:
    libhypre = CDLL("libHYPRE.so")
except OSError:
    print("Error: libHYPRE.so could not be found. Set LD_LIBRARY_PATH.")
    sys.exit(1)

def check_error(ierr, func_name):
    """Helper function for hypre error checking"""
    if ierr != 0:
        print(f"Error in {func_name}: error code {ierr}")
        sys.exit(1)

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # 2. Convert MPI communicator
    # Get memory address from mpi4py communicator and convert to c_void_p.
    # Note: This works for OpenMPI, but MPICH-based implementations might require c_int.
    comm_ptr = MPI._addressof(comm)
    hypre_comm = c_void_p(comm_ptr)

    # 3. Matrix partition settings (Example: Each rank owns 2 rows)
    local_nrows = 2
    ilower = rank * local_nrows
    iupper = (rank + 1) * local_nrows - 1
    
    # Handle (pointer) for IJMatrix
    ij_matrix = c_void_p()

    # -------------------------------------------------------
    # Argument type definitions for hypre functions (Argtypes)
    # Define argument types for main functions for safety
    # -------------------------------------------------------
    
    # HYPRE_IJMatrixCreate(MPI_Comm, ilower, iupper, jlower, jupper, matrix_ptr)
    libhypre.HYPRE_IJMatrixCreate.argtypes = [c_void_p, c_int, c_int, c_int, c_int, POINTER(c_void_p)]
    
    # HYPRE_IJMatrixSetObjectType(matrix, type)
    libhypre.HYPRE_IJMatrixSetObjectType.argtypes = [c_void_p, c_int]
    
    # HYPRE_IJMatrixInitialize(matrix)
    libhypre.HYPRE_IJMatrixInitialize.argtypes = [c_void_p]
    
    # HYPRE_IJMatrixSetValues(matrix, nrows, ncols, rows, cols, values)
    libhypre.HYPRE_IJMatrixSetValues.argtypes = [
        c_void_p, c_int, POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_double)
    ]
    
    # HYPRE_IJMatrixAssemble(matrix)
    libhypre.HYPRE_IJMatrixAssemble.argtypes = [c_void_p]
    
    # HYPRE_IJMatrixDestroy(matrix)
    libhypre.HYPRE_IJMatrixDestroy.argtypes = [c_void_p]

    # -------------------------------------------------------
    # Execution phase
    # -------------------------------------------------------

    # A. Create the matrix
    ierr = libhypre.HYPRE_IJMatrixCreate(
        hypre_comm, 
        c_int(ilower), c_int(iupper), 
        c_int(ilower), c_int(iupper), 
        byref(ij_matrix)
    )
    check_error(ierr, "HYPRE_IJMatrixCreate")

    # Set object type (HYPRE_PARCSR = 5555)
    HYPRE_PARCSR = 5555
    ierr = libhypre.HYPRE_IJMatrixSetObjectType(ij_matrix, c_int(HYPRE_PARCSR))
    check_error(ierr, "HYPRE_IJMatrixSetObjectType")

    # Initialize
    ierr = libhypre.HYPRE_IJMatrixInitialize(ij_matrix)
    check_error(ierr, "HYPRE_IJMatrixInitialize")

    # B. Set values (Simple example setting values on the diagonal)
    # Create and pass ctypes arrays
    
    nrows_to_set = 1 # Example setting 1 row at a time
    ncols_per_row = [1] # Number of non-zero elements per row
    
    # Row indices (Global Index)
    rows_indices = [ilower] 
    # Column indices (Global Index)
    cols_indices = [ilower]
    # Values
    values = [10.0 + rank]

    # Convert to ctypes arrays
    ncols_ptr = (c_int * 1)(*ncols_per_row)
    rows_ptr = (c_int * 1)(*rows_indices)
    cols_ptr = (c_int * 1)(*cols_indices)
    vals_ptr = (c_double * 1)(*values)

    # Pass values to hypre
    ierr = libhypre.HYPRE_IJMatrixSetValues(
        ij_matrix, 
        c_int(nrows_to_set), 
        ncols_ptr, 
        rows_ptr, 
        cols_ptr, 
        vals_ptr
    )
    check_error(ierr, "HYPRE_IJMatrixSetValues")

    # C. Assemble (Synchronization point)
    ierr = libhypre.HYPRE_IJMatrixAssemble(ij_matrix)
    check_error(ierr, "HYPRE_IJMatrixAssemble")

    if rank == 0:
        print("Matrix setup and assembly complete.")

    # Perform solver setup here...

    # D. Destroy
    ierr = libhypre.HYPRE_IJMatrixDestroy(ij_matrix)
    check_error(ierr, "HYPRE_IJMatrixDestroy")

if __name__ == "__main__":
    main()
"""

---

"""python
import sys
import ctypes
from ctypes import POINTER, c_int, c_double, c_void_p, byref, CDLL
from mpi4py import MPI
import numpy as np

# -------------------------------------------------------
# Configuration and Library Loading
# -------------------------------------------------------
try:
    libhypre = CDLL("libHYPRE.so")
except OSError:
    print("Error: libHYPRE.so not found.")
    sys.exit(1)

def check_error(ierr, msg):
    if ierr != 0:
        print(f"Error: {msg} (code {ierr})")
        sys.exit(1)

# -------------------------------------------------------
# NumPy Helper Function (Crucial for Performance)
# -------------------------------------------------------
def as_pointer(numpy_array, c_type):
    """
    Get the memory address of a NumPy array as a ctypes pointer.
    This avoids data copying, ensuring high speed.
    """
    # Ensure the array is contiguous in memory (C-contiguous)
    if not numpy_array.flags['C_CONTIGUOUS']:
        numpy_array = np.ascontiguousarray(numpy_array)
    return numpy_array.ctypes.data_as(POINTER(c_type))

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Convert MPI communicator
    hypre_comm = c_void_p(MPI._addressof(comm))

    # Problem size configuration
    global_n = 1000
    local_n = global_n // size
    ilower = rank * local_n
    iupper = (rank + 1) * local_n - 1
    # The last rank handles the remainder
    if rank == size - 1:
        iupper = global_n - 1
        local_n = iupper - ilower + 1

    # -------------------------------------------------------
    # 1. Create IJMatrix (Using NumPy)
    # -------------------------------------------------------
    ij_matrix = c_void_p()
    libhypre.HYPRE_IJMatrixCreate(hypre_comm, c_int(ilower), c_int(iupper), 
                                  c_int(ilower), c_int(iupper), byref(ij_matrix))
    libhypre.HYPRE_IJMatrixSetObjectType(ij_matrix, c_int(5555)) # HYPRE_PARCSR
    libhypre.HYPRE_IJMatrixInitialize(ij_matrix)

    # Prepare data with NumPy arrays (Example: 1D Laplacian tridiagonal matrix)
    # Number of non-zero elements per row
    ncols = np.full(local_n, 3, dtype=np.int32) 
    
    # Data storage lists (Usually flattened per row, but hypre requires pointers
    # per row, so some manipulation is needed)
    # For simplicity, we implement a loop to set row by row here,
    # but using AddValues in batches is more efficient in production.
    
    values = np.array([[-1.0, 2.0, -1.0]], dtype=np.float64)
    cols = np.array([[ilower-1, ilower, ilower+1]], dtype=np.int32)
    
    # Boundary condition handling (Simplified)
    for i in range(local_n):
        global_row = ilower + i
        row_indices = np.array([global_row], dtype=np.int32)
        
        # Generate column indices and values
        current_cols = np.array([global_row-1, global_row, global_row+1], dtype=np.int32)
        current_vals = np.array([-1.0, 2.0, -1.0], dtype=np.float64)
        
        # Boundary processing
        valid_mask = (current_cols >= 0) & (current_cols < global_n)
        current_cols = current_cols[valid_mask]
        current_vals = current_vals[valid_mask]
        current_ncols = np.array([len(current_cols)], dtype=np.int32)

        # Get pointers and pass them
        libhypre.HYPRE_IJMatrixSetValues(
            ij_matrix, 
            c_int(1), 
            as_pointer(current_ncols, c_int), 
            as_pointer(row_indices, c_int), 
            as_pointer(current_cols, c_int), 
            as_pointer(current_vals, c_double)
        )

    libhypre.HYPRE_IJMatrixAssemble(ij_matrix)
    
    # Get the ParCSR object (to pass to the solver)
    par_a = c_void_p()
    libhypre.HYPRE_IJMatrixGetObject(ij_matrix, byref(par_a))

    # -------------------------------------------------------
    # 2. Create and Set IJVector (RHS b and Solution x)
    # -------------------------------------------------------
    ij_b = c_void_p()
    ij_x = c_void_p()

    # Create vectors [cite: 703]
    libhypre.HYPRE_IJVectorCreate(hypre_comm, c_int(ilower), c_int(iupper), byref(ij_b))
    libhypre.HYPRE_IJVectorSetObjectType(ij_b, c_int(5555)) # HYPRE_PARCSR
    libhypre.HYPRE_IJVectorInitialize(ij_b)

    libhypre.HYPRE_IJVectorCreate(hypre_comm, c_int(ilower), c_int(iupper), byref(ij_x))
    libhypre.HYPRE_IJVectorSetObjectType(ij_x, c_int(5555))
    libhypre.HYPRE_IJVectorInitialize(ij_x)

    # Prepare values with NumPy arrays
    # b = 1.0, x = 0.0 (initial guess)
    indices = np.arange(ilower, iupper + 1, dtype=np.int32)
    values_b = np.full(local_n, 1.0, dtype=np.float64)
    values_x = np.full(local_n, 0.0, dtype=np.float64)

    # Set values: Pass NumPy pointers directly [cite: 711]
    libhypre.HYPRE_IJVectorSetValues(
        ij_b, 
        c_int(local_n), 
        as_pointer(indices, c_int), 
        as_pointer(values_b, c_double)
    )
    
    libhypre.HYPRE_IJVectorSetValues(
        ij_x, 
        c_int(local_n), 
        as_pointer(indices, c_int), 
        as_pointer(values_x, c_double)
    )

    # Assemble [cite: 716]
    libhypre.HYPRE_IJVectorAssemble(ij_b)
    libhypre.HYPRE_IJVectorAssemble(ij_x)

    # Get ParVector objects [cite: 717]
    par_b = c_void_p()
    par_x = c_void_p()
    libhypre.HYPRE_IJVectorGetObject(ij_b, byref(par_b))
    libhypre.HYPRE_IJVectorGetObject(ij_x, byref(par_x))

    # -------------------------------------------------------
    # 3. Solver Configuration (PCG + BoomerAMG)
    # -------------------------------------------------------
    solver = c_void_p()
    precond = c_void_p()

    # A. Create Solver (PCG) [cite: 1040, 3820]
    libhypre.HYPRE_ParCSRPCGCreate(hypre_comm, byref(solver))
    
    # Configure solver parameters
    libhypre.HYPRE_PCGSetMaxIter(solver, c_int(1000)) # Max iterations [cite: 4100]
    libhypre.HYPRE_PCGSetTol(solver, c_double(1e-7))  # Convergence tolerance [cite: 4095]
    libhypre.HYPRE_PCGSetPrintLevel(solver, c_int(2)) # Log output (2=detailed) [cite: 4113]

    # B. Create Preconditioner (BoomerAMG) [cite: 909, 3333]
    libhypre.HYPRE_BoomerAMGCreate(byref(precond))
    
    # Configure AMG parameters (Adjust according to the problem)
    libhypre.HYPRE_BoomerAMGSetCoarsenType(precond, c_int(10)) # HMIS coarsening [cite: 912]
    libhypre.HYPRE_BoomerAMGSetRelaxType(precond, c_int(6))    # Sym G-S/SSOR [cite: 954]
    libhypre.HYPRE_BoomerAMGSetNumSweeps(precond, c_int(1))
    libhypre.HYPRE_BoomerAMGSetMaxIter(precond, c_int(1))      # 1 iteration for preconditioner usage [cite: 3366]

    # C. Set Preconditioner to Solver
    # Pass hypre function pointers (Solve and Setup) here [cite: 1054]
    # ctypes allows passing shared library function attributes directly
    libhypre.HYPRE_PCGSetPrecond(
        solver,
        libhypre.HYPRE_BoomerAMGSolve,
        libhypre.HYPRE_BoomerAMGSetup,
        precond
    )

    # D. Setup and Execute [cite: 776-778]
    libhypre.HYPRE_PCGSetup(solver, par_a, par_b, par_x)
    libhypre.HYPRE_PCGSolve(solver, par_a, par_b, par_x)

    # -------------------------------------------------------
    # Get Results and Finalize
    # -------------------------------------------------------
    
    # Retrieve results back to NumPy array (Copy values from hypre)
    # Call GetValues to update x values
    final_values = np.zeros(local_n, dtype=np.float64)
    libhypre.HYPRE_IJVectorGetValues(
        ij_x, 
        c_int(local_n), 
        as_pointer(indices, c_int), 
        as_pointer(final_values, c_double)
    )

    if rank == 0:
        print("Solve complete.")
        print(f"Sample result (Rank 0, first 5): {final_values[:5]}")

    # Free memory
    libhypre.HYPRE_BoomerAMGDestroy(precond)
    libhypre.HYPRE_ParCSRPCGDestroy(solver)
    libhypre.HYPRE_IJVectorDestroy(ij_x)
    libhypre.HYPRE_IJVectorDestroy(ij_b)
    libhypre.HYPRE_IJMatrixDestroy(ij_matrix)

if __name__ == "__main__":
    main()
"""
