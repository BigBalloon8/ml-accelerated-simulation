# navier_stokes_sim.py
"""
2‑D Incompressible Navier‑Stokes fluid simulation using PyTorch
----------------------------------------------------------------
This script evolves an incompressible velocity field on a regular grid
with *semi‑Lagrangian advection*, *explicit viscosity* and a *Jacobi
projection* step to enforce zero divergence:

u*  = Advect(u)
ū   = u*  + νΔu*  · dt            # diffusion
u^{n+1}= ū − ∇p                  # projection (∇·u^{n+1}=0)

For clarity and interactivity the code is intentionally compact:
* A single convolution kernel expresses the Laplacian and gradients.
* Grid‑sample handles back‑tracing for advection.
* Matplotlib visualises speed magnitude every few steps.

GPU acceleration is automatic when a CUDA device is available.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ─── Simulation parameters ────────────────────────────────────────────────────
H, W         = 128, 128      # grid resolution
DT           = 0.1           # time step
VISCOSITY    = 0.0005        # kinematic viscosity ν
STEPS        = 500           # simulation steps
VIS_EVERY    = 10            # visualisation cadence
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Helper kernels (finite differences) ──────────────────────────────────────
DX = 1.0  # cell size (uniform)
_laplace = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32,
                        device=DEVICE).view(1, 1, 3, 3) / (DX ** 2)
_gradx   = torch.tensor([[-0.5, 0.0, 0.5]], dtype=torch.float32,
                        device=DEVICE).view(1, 1, 1, 3) / DX
_grady   = torch.tensor([[-0.5], [0.0], [0.5]], dtype=torch.float32,
                        device=DEVICE).view(1, 1, 3, 1) / DX


def laplacian(field: torch.Tensor) -> torch.Tensor:
    """5‑point Laplacian Δfield"""
    return F.conv2d(field, _laplace, padding=1)


def gradient(scalar: torch.Tensor) -> torch.Tensor:
    """∇scalar → tensor with two channels (dx, dy)"""
    gx = F.conv2d(scalar, _gradx, padding=(0, 1))
    gy = F.conv2d(scalar, _grady, padding=(1, 0))
    return torch.cat([gx, gy], dim=1)


def divergence(vec: torch.Tensor) -> torch.Tensor:
    """∇·vec (vec has two channels)"""
    gx = F.conv2d(vec[:, 0:1], _gradx, padding=(0, 1))
    gy = F.conv2d(vec[:, 1:2], _grady, padding=(1, 0))
    return gx + gy


# ─── Core fluid operators ─────────────────────────────────────────────────────

def advect(field: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
    """Semi‑Lagrangian advection of *field* by *velocity*"""
    B, C, H_, W_ = field.shape
    y, x = torch.meshgrid(torch.arange(H_, device=DEVICE),
                          torch.arange(W_, device=DEVICE), indexing="ij")
    x = x.float().unsqueeze(0) - DT * velocity[:, 0]
    y = y.float().unsqueeze(0) - DT * velocity[:, 1]
    # Normalize to [-1,1] for grid_sample
    x = (x / (W_ - 1)) * 2 - 1
    y = (y / (H_ - 1)) * 2 - 1
    grid = torch.stack((x, y), dim=-1)  # B × H × W × 2
    return F.grid_sample(field, grid, mode="bilinear",
                         padding_mode="border", align_corners=True)


def diffuse(vec: torch.Tensor) -> torch.Tensor:
    """Explicit viscosity (Forward‑Euler)"""
    return vec + VISCOSITY * laplacian(vec) * DT


def project(vec: torch.Tensor, iters: int = 40) -> torch.Tensor:
    """Jacobi solve of Poisson eqn to enforce ∇·u = 0"""
    p   = torch.zeros(vec.size(0), 1, H, W, device=DEVICE)
    div = divergence(vec)
    for _ in range(iters):
        p = (laplacian(p) - div) * 0.25  # Jacobi iteration
    grad_p = gradient(p)
    return vec - grad_p  # u ← u − ∇p


# ─── Initial condition: a small vortex in the centre ──────────────────────────
vel = torch.zeros(1, 2, H, W, device=DEVICE)
Y, X = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
X, Y = X.to(DEVICE).float(), Y.to(DEVICE).float()
xc, yc = (W - 1) / 2, (H - 1) / 2
r = torch.sqrt((X - xc) ** 2 + (Y - yc) ** 2)
mask = r < 20.0
vel[0, 0][mask] = -(Y[mask] - yc)  # u‑component (tangential)
vel[0, 1][mask] = +(X[mask] - xc)  # v‑component
vel *= 0.01  # scale magnitude

# ─── Time integration loop ────────────────────────────────────────────────────
plt.ion()
fig, ax = plt.subplots(figsize=(5, 5))

for step in range(STEPS):
    # 1. Advect velocity by itself
    vel = advect(vel, vel)

    # 2. Diffusion / viscosity
    vel = diffuse(vel)

    # 3. Projection (pressure solve) to enforce incompressibility
    vel = project(vel)

    # 4. Visualise speed field every VIS_EVERY steps
    if step % VIS_EVERY == 0:
        speed = torch.sqrt((vel[0, 0] ** 2 + vel[0, 1] ** 2)).cpu()
        ax.clear()
        im = ax.imshow(speed, cmap="viridis", origin="lower")
        ax.set_title(f"Step {step}")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.pause(0.001)

plt.ioff()
plt.show()

# ── End of script ─────────────────────────────────────────────────────────────
