
# Physics-Informed Numerical Methods

A collection of computational physics and machine learning tools for solving partial differential equations (PDEs), combining classical numerical methods with modern physics-informed deep learning approaches.

## Projects

| Project                                                 | Method                             | Equation                | Framework     |
| ------------------------------------------------------- | ---------------------------------- | ----------------------- | ------------- |
| [FEM Maxwell Solver](#fem-solver-for-2d-magnetostatics) | Finite Element Method              | Maxwell (magnetostatic) | scikit-fem    |
| [PINN Stokes Flow](#pinn-for-2d-stokes-flow)            | Physics-Informed Neural Network    | Stokes                  | JAX           |
| [Tensor Networks Notebook](#tensor-networks-notebook)   | TT/MPS, SVD/QR, ED (Lanczos), DMRG | —                       | NumPy / SciPy |

---

## FEM Solver for 2D Magnetostatics

Solves Maxwell's equations in the magnetostatic limit using the Finite Element Method.

**Governing Equation:**

$$\nabla^2 A_z = -\mu_0 J_z$$

where $A_z$ is the magnetic vector potential and $J_z$ is the current density.

**Features:**

* Weak form discretization with linear/quadratic elements
* Dirichlet boundary conditions (A = 0)
* Magnetic field recovery via $\mathbf{B} = \nabla \times \mathbf{A}$

```
fem_maxwell/
├── fem_forward_solver.py    # Main FEM implementation
└── README.md
```

**Example:** Wire carrying current

```python
from fem_forward_solver import forward_solve, create_wire_current

J = create_wire_current(nx=64, ny=64, x0=0.5, y0=0.5, radius=0.05, current=1.0)
A, Bx, By = forward_solve(J, dx=0.01, dy=0.01)
```

---

## PINN for 2D Stokes Flow

Solves the incompressible Stokes equations using Physics-Informed Neural Networks, relevant to blood flow in small vessels (low Reynolds number).

**Governing Equations:**

$$-\nabla p + \mu \nabla^2 \mathbf{u} = 0$$

$$\nabla \cdot \mathbf{u} = 0$$

**Problem Setup:**

```
       Inlet (x = 0)
       u = u(y), v = 0
             │
             ▼
┌───────────────────────┐
│  Wall: u = v = 0      │
│                       │
│   ──► ──► Flow ──►    │
│                       │
│  Wall: u = v = 0      │
└───────────────────────┘
             │
             ▼
       Outlet (x = L)
       ∂p/∂x = 0
```

**Features:**

* Automatic differentiation for PDE residuals
* GPU-accelerated training via `jax.jit` and `jax.vmap`
* Validation against analytical Poiseuille solution

```
pinn_stokes/
├── stokes_channel_pinn.py   # Main PINN implementation
└── README.md
```

**Example:**

```python
from stokes_channel_pinn import Config, train
import jax.random as random

cfg = Config(L=2.0, H=0.5, mu=1.0, n_epochs=10000)
params, history = train(cfg, random.PRNGKey(42))
```

---

## Tensor Networks Notebook

A self-contained Jupyter notebook, **`tensor_networks.ipynb`**, with runnable code and notes covering tensor network fundamentals and core algorithms commonly used in computational physics.

**Covers:**

* Tensor basics: contraction, reshaping, Frobenius norm
* Matrix factorizations for tensors:

  * SVD and optimal truncation (Eckart–Young)
  * QR decomposition and isometries
* Tensor network structure and gauge:

  * Tree tensor networks (loop-free graphs)
  * Gauge freedom (inserting $X X^{-1}$ on internal bonds)
  * Centers of orthogonality (QR “pulling-through”)
  * Direct orthogonalization (branch density matrices $\rho$ and $\rho^{1/2}$)
* Canonical forms (MPS/TT ideas, link-centered singular values)
* **Exact Diagonalization (Lanczos)** using matrix-free $|\psi\rangle \mapsto H|\psi\rangle$
* **Finite DMRG (open boundaries)** with two-site updates, environments, and SVD truncation

**Typical use cases:**

* High-dimensional compression (TT/MPS)
* Ground-state search for 1D quantum systems (DMRG)
* Small-system validation (ED/Lanczos)

---

## Installation

```bash
git clone https://github.com/tyy0811/physics-informed-numerical-methods.git
cd physics-informed-numerical-methods

# Install dependencies
pip install -r requirements.txt
```

**Dependencies:**

| Project                  | Requirements                                                       |
| ------------------------ | ------------------------------------------------------------------ |
| FEM Maxwell              | `scikit-fem`, `scipy`, `numpy`, `matplotlib`                       |
| PINN Stokes              | `jax`, `jaxlib`, `optax`, `matplotlib`                             |
| Tensor Networks Notebook | `numpy`, `scipy`, `opt_einsum` (optional), `matplotlib` (optional) |

> To run the notebook: `jupyter lab` or `jupyter notebook`.

---

## Repository Structure

```
physics-informed-numerical-methods/
├── README.md
├── requirements.txt
├── tensor_networks.ipynb
├── fem_maxwell/
│   ├── fem_forward_solver.py
│   └── README.md
├── pinn_stokes/
│   ├── stokes_channel_pinn.py
│   └── README.md
```

---

## Methods Overview

### Classical vs. Data-Driven Approaches

| Aspect         | FEM                   | PINN                           |
| -------------- | --------------------- | ------------------------------ |
| Discretization | Mesh-based            | Mesh-free (collocation points) |
| Derivatives    | Weak form integration | Automatic differentiation      |
| Scalability    | Sparse linear systems | GPU-parallelized training      |
| Flexibility    | Fixed geometry        | Easy geometry changes          |

### When to Use What

* **FEM**: Well-posed problems, need guaranteed accuracy, complex material properties
* **PINN**: Inverse problems, surrogate modeling, when mesh generation is difficult
* **Tensor Networks**: High-dimensional problems, data compression, quantum-inspired ML, 1D quantum ground states (DMRG)

---
