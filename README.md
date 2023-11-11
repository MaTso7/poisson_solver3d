# poisson_solver3d
## A parallel solver for the 3D Poisson equation.

The program `poisson_solver3d.c` solves the 3D Poisson equation
$$-\Delta u = f(x), \quad x\in D$$
on a box $D$ with Dirichlet boundary conditions
$$u(x) = g(x), \quad x\in\partial D$$
for given functions $f$, $g$. For this, $D$ is discretized using a uniform step size along all three dimensions.

The following algorithms are available:
- Jacobi,
- Gauß-Seidel,
- SOR (successive over-relaxation),
- CG (conjugate gradients).

All algorithms are parallelized with MPI using domain decomposition.

The MATLAB script `plot_poisson.m` can be used to read the file `result` and plot the result.

### Note
This program was written as an assignment during my Master's. I hope someone finds it useful.
