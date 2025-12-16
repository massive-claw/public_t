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
