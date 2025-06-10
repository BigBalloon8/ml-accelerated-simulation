import torch
from torch_cfd import grids, boundaries
from torch_cfd.initial_conditions import filtered_velocity_field

from torch_cfd.equations import stable_time_step
from torch_cfd.fvm import RKStepper, NavierStokes2DFVMProjection
from torch_cfd.forcings import KolmogorovForcing
import torch_cfd.finite_differences as fdm

from tqdm import tqdm

import seaborn as sns
import xarray
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

n = 256
batch_size = 4
density = 1.0
max_velocity = 3.0
peak_wavenumber = 4.0
cfl_safety_factor = 0.5
viscosity = 1e-3
print(f"Cuda Available: {torch.cuda.is_available()}")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)
inner_steps = 20
outer_steps = 100
diam = 2 * torch.pi


with open("test.txt", "w") as f:
    pass

grid = grids.Grid((n, n), domain=((0, diam), (0, diam)), device=device)

v0 = filtered_velocity_field(
    grid, max_velocity, peak_wavenumber, iterations=3, random_state=41,
    device=device, batch_size=batch_size,)
print(f"Initial velocity field:\n{v0.shape} {v0.dtype} {v0.device}")
v0div = fdm.divergence(v0)
pressure_bc = boundaries.get_pressure_bc_from_velocity(v0)

print(f"divergence of initial velocity L2: {torch.linalg.norm(v0div).data:.2e}")

dt = stable_time_step(
    dx=min(grid.step),
    max_velocity=max_velocity,
    max_courant_number=cfl_safety_factor,
    viscosity=viscosity,
)
print(f"dt: {dt} | batch size: {batch_size} | grid: {grid.shape}")
step_fn = RKStepper.from_method(method="classic_rk4", requires_grad=False, dtype=torch.float64)
forcing_fn = KolmogorovForcing(diam=diam, wave_number=int(peak_wavenumber),
    grid=grid, offsets=(v0[0].offset, v0[1].offset))

ns2d = NavierStokes2DFVMProjection(
    viscosity=viscosity,
    grid=grid,
    bcs=(v0[0].bc, v0[1].bc),
    density=density,
    drag=0.1,
    forcing=forcing_fn,
    solver=step_fn,
    # set_laplacian=False,
).to(v0.device)


v = v0
trajectory = [[v0[0].data.detach().cpu().numpy()], [v0[1].data.detach().cpu().numpy()]]
nan_count = 0
print(v0.shape)


print(v.shape)
with tqdm(total=outer_steps*inner_steps) as pbar:
    with torch.no_grad():
        for i in range(outer_steps):
            for j in range(inner_steps):
                v, p = step_fn.forward(v, dt, equation=ns2d)
                if torch.isnan(v[0].data).any():
                    print(f"NaN detected at {i*inner_steps + j}")
                    nan_count += 1
                    break
            if nan_count > 0:
                break
            trajectory[0].append(v[0].data.detach().cpu().numpy())
            trajectory[1].append(v[1].data.detach().cpu().numpy())
            #print(torch.stack(v.data).shape)
            #input()
            pbar.update(inner_steps)


# GRAPHING

idxes = np.random.choice(np.arange(batch_size), size=1, replace=False)

print(trajectory[0][0].shape)
trajectory_plot = np.stack(trajectory).astype(np.float64)

print(trajectory_plot.shape)
for idx in idxes:
    coords={
            "time": dt * inner_steps * np.arange(outer_steps),
            "x": np.linspace(0, 2 * np.pi, n),
            "y": np.linspace(0, 2 * np.pi, n),
        }
    ds = xarray.Dataset(
        {
            "u": (("time", "x", "y"), trajectory_plot[0, 1:, idx, ...]),
            "v": (("time", "x", "y"), trajectory_plot[1, 1:, idx, ...]),
        },
        coords=coords,
    )


    def vorticity(ds):
        return (ds.v.differentiate("x") - ds.u.differentiate("y")).rename(f"vorticity of sample {idx}")

    
    ds.pipe(vorticity).thin(time=20).plot.imshow(col="time", cmap=sns.cm.icefire, robust=True, col_wrap=5)
    plt.show()

    # Animation
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    vort = ds.pipe(vorticity)
    image = vort.isel(time=0).plot.imshow(ax=ax, cmap=sns.cm.icefire, robust=True)
    freq = 2
    def update(frame):
        image.set_array(vort.isel(time=frame*freq).values)
        ax.set_title(f"Vorticity at time {ds.time[frame*freq].values:.2f} s")
        #vort.isel(time=frame).plot.imshow(ax=ax, cmap=sns.cm.icefire, robust=True, add_colorbar= False)
    ani = animation.FuncAnimation(fig, update, frames=len(ds.time)//freq, repeat=False, blit=False, interval=5)
    writervideo = animation.FFMpegWriter(fps=60)
    ani.save('sim.mp4', writer=writervideo)
    plt.show()

    
    