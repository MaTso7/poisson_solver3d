# poisson_solver3d

A parallel C/MPI solver for the three-dimensional Poisson equation using finite differences and domain decomposition.

This project solves

```math
-\Delta u = f \quad \text{in } D,
\qquad
u = g \quad \text{on } \partial D,
```

on a rectangular 3D domain with Dirichlet boundary conditions. The domain is discretized on a uniform Cartesian grid and distributed across MPI processes using a three-dimensional Cartesian process topology.

The project was originally developed during my Master's studies and is kept here as a compact example of numerical PDE solving with distributed-memory parallelization.

## Features

* Finite-difference discretization of the 3D Poisson equation
* MPI-based domain decomposition
* Cartesian MPI communicator with halo exchange
* Iterative solvers: Jacobi, Gauss-Seidel, SOR, and Conjugate Gradient
* Binary output for post-processing
* MATLAB/Octave script for visualization and validation

## Build

Requires an MPI implementation such as MPICH or OpenMPI.

```bash
mpicc -O3 poisson_solver3d.c -lm -o poisson_solver3d
```

## Usage

Run the program with `mpirun` or `mpiexec`. 

Example:

```bash
mpirun -np 4 ./poisson_solver3d \
  alg=cg \
  m=100 n=20 q=10 \
  h=0.1 \
  threshold=1e-8 \
  x=0 y=0 z=0
```

The arguments are:

| Argument      | Meaning                                               |
| ------------- | ----------------------------------------------------- |
| `alg`         | Solver algorithm: `jacobi`, `gs`, `sor`, or `cg`      |
| `m`, `n`, `q` | Number of grid points in the three spatial dimensions |
| `h`           | Uniform grid spacing                                  |
| `threshold`   | Stopping criterion                                    |
| `x`, `y`, `z` | Starting coordinates of the computational domain      |

The solver writes the numerical solution to a binary file named `results`.

## Parallelization

The computational domain is decomposed into subdomains and distributed across MPI processes. Neighboring processes exchange halo data during the iterative solve. MPI subarray datatypes are used for non-contiguous boundary regions.

## Validation and visualization

The default test problem uses the analytical reference solution

```math
u(x,y,z) = (1 + x + z)\sin(x+y).
```

The corresponding right-hand side and boundary data are defined in `poisson_solver3d.c`.

After running the solver, use the MATLAB script to visualize the result:

```matlab
plot_poisson
```

The script reads `results`, compares the numerical solution to the analytical reference solution, and plots the relative difference.
The parameters in `plot_poisson.m` should match the parameters used when running the solver.


## Scope and limitations

This is a compact scientific-computing example rather than a general-purpose PDE solver library. The right-hand side and boundary function are currently defined in the source code, and the focus is on demonstrating numerical methods and MPI-based parallelization.

