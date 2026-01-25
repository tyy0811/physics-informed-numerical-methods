# -*- coding: utf-8 -*-
"""
FEM Forward Solver for 2D Magnetostatics using scikit-fem
==========================================================

Solves Maxwell's equations in the magnetostatic limit using the 
Finite Element Method (FEM) with scikit-fem.

Strong form:
    Laplacian(A_z) = -mu_0 * J_z

Weak form:
    integral( grad(A) . grad(v) dx ) = mu_0 * integral( J * v dx )

Where:
- A_z is the magnetic vector potential (z-component)
- J_z is the current density (perpendicular to the 2D plane)
- mu_0 is the permeability of free space
- v is the test function

The magnetic field is computed via B = curl(A):
    B_x = dA_z/dy
    B_y = -dA_z/dx

Installation:
    pip install scikit-fem """



import numpy as np
import sys
import subprocess
from typing import Tuple

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    from skfem import *
    from skfem.models.poisson import laplace, unit_load
    from skfem.helpers import grad, dot
    SKFEM_AVAILABLE = True
except ImportError:
    print("scikit-fem not found. Installing...")
    try:
        install_package("scikit-fem")
        from skfem import *
        from skfem.models.poisson import laplace, unit_load
        from skfem.helpers import grad, dot
        SKFEM_AVAILABLE = True
        print("scikit-fem installed successfully!")
    except Exception as e:
        SKFEM_AVAILABLE = False
        print(f"Failed to install scikit-fem: {e}")


from scipy.sparse.linalg import spsolve

# Physical constants
MU_0 = 4 * np.pi * 1e-7  # Permeability of free space [H/m]


class MagnetostaticSolver:
    """
    scikit-fem based solver for 2D magnetostatics.
    
    Solves the Poisson equation for the magnetic vector potential:
        -Laplacian(A) = mu_0 * J
    
    with Dirichlet boundary conditions A = 0 on the domain boundary.
    """
    
    def __init__(self, nx: int = 64, ny: int = 64, 
                 Lx: float = 0.1, Ly: float = 0.1,
                 degree: int = 1):
        """
        Initialize the magnetostatic solver.
        
        Parameters
        ----------
        nx, ny : int
            Number of mesh elements in x and y directions
        Lx, Ly : float
            Physical domain size [m]
        degree : int
            Polynomial degree for finite elements (1 = linear, 2 = quadratic)
        """
        
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.dx_grid = Lx / nx
        self.dy_grid = Ly / ny
        
        # Create mesh - rectangular mesh scaled to physical dimensions
        self.mesh = MeshTri.init_tensor(
            np.linspace(0, Lx, nx + 1),
            np.linspace(0, Ly, ny + 1)
        )
        
        # Define element type based on degree
        self.element = ElementTriP2() if degree == 2 else ElementTriP1()
        
        # Create basis
        self.basis = Basis(self.mesh, self.element)
        
        # Find boundary DOFs for Dirichlet BC
        epsilon = 1e-10
        self.boundary_dofs = self.basis.get_dofs(
            lambda x: (x[0] < epsilon) | (x[0] > Lx - epsilon) | 
                      (x[1] < epsilon) | (x[1] > Ly - epsilon)
        ).all()
        
    def solve_from_array(self, J_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve the magnetostatic problem from a numpy array of current density.
        
        Parameters
        ----------
        J_array : np.ndarray
            Current density field of shape (ny, nx) [A/m^2]
        
        Returns
        -------
        A : np.ndarray
            Magnetic vector potential of shape (ny, nx)
        Bx, By : np.ndarray
            Magnetic field components of shape (ny, nx)
        """
        from scipy.interpolate import RegularGridInterpolator
        
        # Create interpolator from the array
        x_grid = np.linspace(0, self.Lx, self.nx)
        y_grid = np.linspace(0, self.Ly, self.ny)
        
        # RegularGridInterpolator expects (y, x) ordering for the data
        interp = RegularGridInterpolator((y_grid, x_grid), J_array, 
                                          bounds_error=False, fill_value=0.0)
        
        # Define the source term (right-hand side)
        @LinearForm
        def source_functional(v, w):
            # Evaluate J at quadrature points
            points = np.column_stack([w.x[1].flatten(), w.x[0].flatten()])
            J_vals = interp(points).reshape(w.x[0].shape)
            return MU_0 * J_vals * v
        
        # Assemble stiffness matrix (Laplacian)
        @BilinearForm
        def stiffness(u, v, w):
            return dot(grad(u), grad(v))
        
        K = stiffness.assemble(self.basis)
        
        # Assemble load vector
        f = source_functional.assemble(self.basis)
        
        # Apply Dirichlet boundary conditions (A = 0 on boundary)
        K_bc, f_bc = enforce(K, f, D=self.boundary_dofs)
        
        # Solve the linear system
        A_solution = spsolve(K_bc, f_bc)
        
        # Compute magnetic field B = curl(A)
        # B_x = dA/dy, B_y = -dA/dx
        Bx_array, By_array = self._compute_magnetic_field(A_solution)
        
        # Interpolate to regular grid
        A_array = self._solution_to_grid(A_solution)
        
        return A_array, Bx_array, By_array
    
    def _compute_magnetic_field(self, A_solution: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute magnetic field components from vector potential.
        
        B_x = dA/dy, B_y = -dA/dx
        """
        # Interpolate A to regular grid first, then compute gradients
        A_grid = self._solution_to_grid(A_solution)
        
        # Compute grid spacing
        dx = self.Lx / (self.nx - 1) if self.nx > 1 else self.Lx
        dy = self.Ly / (self.ny - 1) if self.ny > 1 else self.Ly
        
        # B_x = dA/dy (central differences in interior, forward/backward at boundaries)
        Bx = np.zeros_like(A_grid)
        Bx[1:-1, :] = (A_grid[2:, :] - A_grid[:-2, :]) / (2 * dy)
        Bx[0, :] = (A_grid[1, :] - A_grid[0, :]) / dy
        Bx[-1, :] = (A_grid[-1, :] - A_grid[-2, :]) / dy
        
        # B_y = -dA/dx
        By = np.zeros_like(A_grid)
        By[:, 1:-1] = -(A_grid[:, 2:] - A_grid[:, :-2]) / (2 * dx)
        By[:, 0] = -(A_grid[:, 1] - A_grid[:, 0]) / dx
        By[:, -1] = -(A_grid[:, -1] - A_grid[:, -2]) / dx
        
        return Bx, By
    
    def _solution_to_grid(self, solution: np.ndarray) -> np.ndarray:
        """
        Interpolate FEM solution to regular grid.
        """
        from scipy.interpolate import LinearNDInterpolator
        
        # Get nodal coordinates
        coords = self.mesh.p.T  # (n_nodes, 2)
        
        x_grid = np.linspace(0, self.Lx, self.nx)
        y_grid = np.linspace(0, self.Ly, self.ny)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Handle solution size mismatch (P2 elements have more DOFs)
        if len(solution) != len(coords):
            # For P2 elements, use basis interpolation
            result = np.zeros((self.ny, self.nx))
            for i in range(self.ny):
                for j in range(self.nx):
                    try:
                        result[i, j] = self.basis.interpolator(solution)(
                            np.array([[x_grid[j]], [y_grid[i]]])
                        )[0]
                    except:
                        result[i, j] = 0.0
            return result
        
        # For P1 elements, use linear interpolation from mesh nodes
        interp = LinearNDInterpolator(coords, solution, fill_value=0.0)
        return interp(X, Y)


def forward_solve(J: np.ndarray, dx: float = 0.01, dy: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Complete forward solve: J -> B using FEM (scikit-fem).
    
    Given current density J, compute magnetic field B.
    
    Parameters
    ----------
    J : np.ndarray
        Current density field of shape (ny, nx) [A/m^2]
    dx, dy : float
        Grid spacing [m], default 1 cm
    
    Returns
    -------
    A : np.ndarray
        Magnetic vector potential
    Bx, By : np.ndarray
        Magnetic field components
    """
    if not SKFEM_AVAILABLE:
        raise ImportError("scikit-fem is required. Install via: pip install scikit-fem")
        
    ny, nx = J.shape
    Lx = nx * dx
    Ly = ny * dy
    
    solver = MagnetostaticSolver(nx=nx, ny=ny, Lx=Lx, Ly=Ly)
    A, Bx, By = solver.solve_from_array(J)
    
    return A, Bx, By


def create_wire_current(nx: int, ny: int, x0: float, y0: float, 
                        radius: float, current: float) -> np.ndarray:
    """
    Create a current density distribution for a wire (circular cross-section).
    
    Parameters
    ----------
    nx, ny : int
        Grid dimensions
    x0, y0 : float
        Wire center position (normalized 0-1)
    radius : float
        Wire radius (normalized 0-1)
    current : float
        Total current [A] (distributed uniformly)
    
    Returns
    -------
    J : np.ndarray
        Current density field of shape (ny, nx)
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    
    # Distance from wire center
    r = np.sqrt((X - x0)**2 + (Y - y0)**2)
    
    # Wire cross-sectional area
    area = np.pi * radius**2
    
    # Current density (uniform inside wire)
    J = np.where(r <= radius, current / area, 0.0)
    
    return J


def create_gaussian_current(nx: int, ny: int, x0: float, y0: float,
                            sigma: float, amplitude: float) -> np.ndarray:
    """
    Create a Gaussian current density distribution.
    
    Parameters
    ----------
    nx, ny : int
        Grid dimensions
    x0, y0 : float
        Center position (normalized 0-1)
    sigma : float
        Standard deviation (normalized 0-1)
    amplitude : float
        Peak current density [A/m^2]
    
    Returns
    -------
    J : np.ndarray
        Current density field of shape (ny, nx)
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    
    J = amplitude * np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
    
    return J


def test_solver():
    """
    Test the scikit-fem FEM solver with a simple wire configuration.
    """
    if not SKFEM_AVAILABLE:
        print("scikit-fem not available. Please install via:")
        print("  pip install scikit-fem")
        return
    
    import matplotlib.pyplot as plt
    
    print("Testing scikit-fem FEM Solver...")
    
    # Grid parameters
    nx, ny = 64, 64
    Lx, Ly = 0.1, 0.1  # 10 cm x 10 cm domain
    dx, dy = Lx / nx, Ly / ny
    
    # Create a wire at the center
    J = create_wire_current(nx, ny, 0.5, 0.5, 0.05, 1.0)
    
    # Solve using scikit-fem
    print("Solving with scikit-fem FEM...")
    A, Bx, By = forward_solve(J, dx, dy)
    B_magnitude = np.sqrt(Bx**2 + By**2)
    
    print("Solution computed!")
    print("  Grid: {} x {}".format(nx, ny))
    print("  Max |B|: {:.2e} T".format(B_magnitude.max()))
    print("  Max A: {:.2e} T*m".format(A.max()))
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    extent = [0, Lx * 100, 0, Ly * 100]  # Convert to cm
    
    # Current density
    im0 = axes[0, 0].imshow(J, origin='lower', extent=extent, cmap='hot')
    axes[0, 0].set_title('Current Density J [A/m^2]')
    axes[0, 0].set_xlabel('x [cm]')
    axes[0, 0].set_ylabel('y [cm]')
    plt.colorbar(im0, ax=axes[0, 0])
    
    # Vector potential
    im1 = axes[0, 1].imshow(A, origin='lower', extent=extent, cmap='viridis')
    axes[0, 1].set_title('Vector Potential A [T*m]')
    axes[0, 1].set_xlabel('x [cm]')
    axes[0, 1].set_ylabel('y [cm]')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Magnetic field magnitude
    im2 = axes[1, 0].imshow(B_magnitude, origin='lower', extent=extent, cmap='plasma')
    axes[1, 0].set_title('Magnetic Field |B| [T]')
    axes[1, 0].set_xlabel('x [cm]')
    axes[1, 0].set_ylabel('y [cm]')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Magnetic field vectors (streamplot)
    x = np.linspace(0, Lx * 100, nx)
    y = np.linspace(0, Ly * 100, ny)
    axes[1, 1].streamplot(x, y, Bx, By, density=1.5, color=B_magnitude, cmap='plasma')
    axes[1, 1].set_title('Magnetic Field Lines')
    axes[1, 1].set_xlabel('x [cm]')
    axes[1, 1].set_ylabel('y [cm]')
    axes[1, 1].set_xlim(0, Lx * 100)
    axes[1, 1].set_ylim(0, Ly * 100)
    axes[1, 1].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('fem_solver_test.png', dpi=150)
    plt.show()
    
    print("\nscikit-fem FEM Solver test completed!")
    print("Output saved to 'fem_solver_test.png'")


if __name__ == "__main__":
    test_solver()
